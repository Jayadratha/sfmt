import os, sys
sys.path.append(sys.path[0]+"/../")
import shutil
import pymeshlab
# import gc
import cv2
import random
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
import wandb
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse, time, contextlib, itertools
from data_utils.multi_loader import MultiDataset
from data_utils.amass_loader import AmassDataset
from data_utils.rignet_loader import RignetDataset
from data_utils.mixamo_loader_vivek import MixamoValidationDataset
from models.networks import PerPartEncoderTpl, PerPartDecoder, HandlePredictorSWTpl
from models.prior import aggregate_window
from models.smpl import SMPL2Mesh
from utils.training import LossManager
from utils.pyrenderer import Renderer
from utils.visualization import visualize_handle, visualize_part
from utils.o3d_wrapper import Mesh
from utils.geometry import arap_loss, fps, get_normal_batch
from models.ops import *
from global_var import *
import numpy as np

parser=argparse.ArgumentParser(description="SFMT")
parser.add_argument("--window",type=int,default=5,help="window size (number of previous predicted frame encodings used as prior); 0 = baseline (no prior)")
parser.add_argument("--agg",type=str,default='weighted_avg',choices=['avg','weighted_avg'],help="how the window of previous encodings is aggregated")
parser.add_argument("--epochs",type=int,default=10,help="number of training epochs")
parser.add_argument("--val_every",type=int,default=12000,help="run validation every N iterations (0 = never)")
parser.add_argument("--ckpt_every",type=int,default=1200,help="write latest.pth every N iterations (must be a multiple of 60) and at the end of each epoch; a run interrupted mid-epoch resumes from the last one")
parser.add_argument("--workers",type=int,default=4,help="data loader worker processes (0 = load in the main process)")
parser.add_argument("--device",type=str,default='cuda:0',help="torch device (cpu is only useful for profiling)")
parser.add_argument("--profile",type=int,default=0,help="time N iterations per phase (after 5 warm-up iterations), print a table and exit")
parser.add_argument("--profile_ops",action='store_true',help="with --profile: also print the torch.profiler operator table")
parser.add_argument("--skip_dead_forward",action='store_true',help="skip the predictor forward on data3, whose output is never used in the loss (only changes the BatchNorm running statistics slightly)")
parser.add_argument("--detect_anomaly",action='store_true',help="torch.autograd anomaly detection (slow, debugging only)")
parser.add_argument("--max_iters",type=int,default=0,help="stop each epoch after N iterations (0 = full epoch; use for smoke tests)")
parser.add_argument("--seed",type=int,default=None,help="seed for model initialisation AND the training data schedule (character draws); unset = the seeds used by the first six runs")
parser.add_argument("--exp_name",type=str,default=None,help="experiment name (default derived from --agg/--window)")
parser.add_argument("--wandb",type=str,default='offline',choices=['online','offline','disabled'],help="wandb mode")
parser.add_argument("--save_ply",type=str,default='last',choices=['none','last','all'],help="dump validation meshes (.ply, ~0.5 GB per validation pass): never / only in the last epoch / every validation")

class PhaseTimer:
    """Per-phase wall-clock timer (synchronises the GPU around each phase, so only enable it for profiling)."""
    def __init__(self, enabled, device):
        self.enabled, self.dev, self.t = enabled, device, {}
    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        if self.dev.type == 'cuda': torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.dev.type == 'cuda': torch.cuda.synchronize(self.dev)
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
    def report(self, n, wall):
        print(f"\n=== per-iteration time over {n} iterations (wall {wall / n * 1000:.0f} ms/it, device-synchronised) ===")
        tot = sum(v for k, v in self.t.items() if not k.startswith('>'))
        for k, v in sorted(self.t.items(), key=lambda kv: (kv[0].lstrip('>') if kv[0].startswith('>') else '~', -kv[1]) if False else -kv[1]):
            print(f"  {k:<16s}{v / n * 1000:8.1f} ms  {100 * v / wall:5.1f}%")
        print(f"  {'(unaccounted)':<16s}{(wall - tot) / n * 1000:8.1f} ms  {100 * (wall - tot) / wall:5.1f}%")


def _worker_init(_):
    torch.set_num_threads(1)


def timed_iter(it, T):
    it = iter(it)
    while True:
        with T('fetch'):
            try:
                item = next(it)
            except StopIteration:
                return
        yield item


def visualize_data(vs, fs, save_path):
    renderer = Renderer(400)
    imgs = []
    for v, f in zip(vs, fs):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(f, list):
            f = f[0][0]
        imgs.append(renderer(v, f))
    imgs = np.concatenate(imgs, 1)
    cv2.imwrite(save_path, imgs)


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(RANDOM_SEED)
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""
Shape and Handle in the same encoder
"""
def main():
    global args
    args=parser.parse_args()
    # default (--seed unset) = the seeds of the first six runs: model init RANDOM_SEED, data schedule 2023
    data_seed = 2023 if args.seed is None else args.seed
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if cuda_avail: torch.cuda.manual_seed_all(args.seed)

    bs = 2
    lr = 1e-4
    exp_name = args.exp_name or (f"baseline" if args.window == 0 else f"{args.agg}_w{args.window}")
    exp_dir = f"{LOG_DIR}/{exp_name}"

    wandb.init(project="MotionTransfer", name=f"motion_60_{exp_name}", mode=args.wandb, config=vars(args))
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)
    ckpt_path = os.path.join(exp_dir, "latest.pth")
   
    smpl = SMPL2Mesh(SMPLH_PATH)
    num_workers = args.workers
    loader_kw = dict(batch_size=bs, shuffle=False, pin_memory=False, drop_last=True, num_workers=num_workers,
                     **(dict(prefetch_factor=6, worker_init_fn=_worker_init) if num_workers > 0 else {}))
    part_aug_scale = ((0.5, 3), (0.7, 1), (0.5, 1.5))
    train_set = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, part_aug_scale=part_aug_scale,
                             part_augmentation=(False, False, False), prob=(0, 1, 0),
                             single_part=False, simplify=True, new_rignet=True, seed=data_seed)
    train_loader = DataLoader(train_set, **loader_kw)      # stateless schedule: sample 2k, 2k+1 -> iteration k (see MultiDataset)
    
    train_set2 = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, single_part=True,
                              part_aug_scale=part_aug_scale,
                              part_augmentation=(False, False, False), prob=(0, 1, 0),
                              preload=train_set.database(), simplify=True, new_rignet=True, seed=data_seed)
    train_loader2 = DataLoader(train_set2, **loader_kw)

    test_mixamo_set = MixamoValidationDataset(MIXAMO_SIMPLIFY_PATH)
    test_mixamo_loader = DataLoader(test_mixamo_set, batch_size=1,
                                    shuffle=False, pin_memory=False, drop_last=False,
                                    num_workers=num_workers)

    

    num_handle = 40
    encoder_shape = PerPartEncoderTpl(3+3, 128)
    predictor = HandlePredictorSWTpl(3+3, num_handle)
    decoder = PerPartDecoder(128*3+7)
 
    device = torch.device(args.device)
    T = PhaseTimer(args.profile > 0, device)
    encoder_shape.to(device)
    predictor.to(device)
    decoder.to(device)
   

    wandb.watch(encoder_shape)
    wandb.watch(predictor)
    wandb.watch(decoder)

    ref_bm_path = os.path.join(SMPLH_PATH, 'neutral/model.npz')
    ref_bm = np.load(ref_bm_path)
    ref_mesh = Mesh(v=ref_bm['v_template'], f=ref_bm['f'])

    optimizer = optim.AdamW(list(encoder_shape.parameters()) +
                            list(decoder.parameters()) +
                            list(predictor.parameters()),
                            lr=lr, foreach=True)
    criterion = torch.nn.L1Loss()

    # (image rendering via pyrender is disabled in the validation loop, so no Renderer is created here)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        encoder_shape.load_state_dict(checkpoint['encoder_shape'])
        decoder.load_state_dict(checkpoint['decoder'])
        predictor.load_state_dict(checkpoint['predictor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('mid_epoch'):          # interrupted inside an epoch: continue from the saved iteration
            start_epoch, resume_iter = checkpoint['epoch'], checkpoint['iter']
            torch.set_rng_state(checkpoint['torch_rnd']); np.random.set_state(checkpoint['numpy_rnd'])
        else:
            start_epoch, resume_iter = checkpoint['epoch'] + 1, 0
        print(f"Successfully load {ckpt_path} (start epoch {start_epoch}, iteration {resume_iter})")
    else:
        start_epoch, resume_iter = 0, 0

    
    loss_m = LossManager(['loss1', 'loss_arap', 'loss_fps', 'loss_sw', 'loss_sw_sim'], writer=writer, epoch=start_epoch)
    mixamo_loss_m = LossManager(['mixamo_dist', 'mixamo_loss'], writer=writer, epoch=start_epoch)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    assert args.ckpt_every % 60 == 0, "--ckpt_every must be a multiple of 60 (sequence length)"
    inp_iter=args.window
    n_iter=len(train_loader)
    print("train_loader",len(train_loader))
    for i_epoch in range(start_epoch, args.epochs):
        encoder_shape.train()
        decoder.train()
        predictor.train()
        wandb_log = {}
        loss_m.update_epoch(i_epoch)
        mixamo_loss_m.update_epoch(i_epoch)
        print(f"=====Experiment {exp_name}\tEpoch {i_epoch}=====")
        
        first_iter, resume_iter = resume_iter, 0
        train_set.start_iter = train_set2.start_iter = first_iter     # workers are (re)started with the iterator below
        second = train_loader2 if not args.skip_dead_forward else itertools.repeat((None, None))
        for i_iter, ((data1, data2), (data3, _)) in enumerate(timed_iter(zip(train_loader, second), T), start=first_iter):
            if args.profile:                                   # --profile: time iterations [5, 5+N) then report and exit
                if i_iter == first_iter + 5:
                    T.t.clear(); t_prof = time.perf_counter()
                    if args.profile_ops:
                        from torch.profiler import profile, ProfilerActivity
                        prof = profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else []))
                        prof.__enter__()
                elif i_iter == first_iter + 5 + args.profile:
                    if device.type == 'cuda': torch.cuda.synchronize()
                    wall = time.perf_counter() - t_prof
                    if args.profile_ops:
                        prof.__exit__(None, None, None)
                        key = 'cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'
                        print(prof.key_averages().table(sort_by=key, row_limit=30, max_name_column_width=60))
                    T.report(args.profile, wall)
                    if device.type == 'cuda': print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")
                    sys.exit(0)
            
            optimizer.zero_grad()
            with T('to_device'):
                for d_ in (data1, data2, data3):
                    if d_ is not None:
                        d_.ptr_list = d_.ptr.tolist()   # per-graph vertex offsets as python ints (avoids GPU syncs later)
                        d_.to(device)
            # f11=data1.triangle[0][0]
            # f21=data2.triangle[0][0]
            # f12=data1.triangle[1][0]
            # f22=data2.triangle[1][0]
            # f31=data3.triangle[0][0]
            # num1=np.max(f11)
            # num2=np.max(f21)
            # num3=np.max(f31)
            

            with T('predictor'):
                hm1, hd_pos1, _, region_score1 = predictor(torch.cat((data1.aug_v0, data1.feat0), 1),
                                                           data=data1, verbose=True)
                _, _, _, region_score2_noaug = predictor(torch.cat((data2.v0, data2.feat0), 1),
                                                         data=data2, verbose=True)
                hm2, hd_pos2, _, region_score2 = predictor(torch.cat((data2.aug_v0, data2.feat0), 1),
                                                           data=data2, verbose=True)
                if not args.skip_dead_forward:      # output unused by the loss; kept (without autograd) for the BatchNorm running stats
                    with torch.no_grad():
                        predictor(torch.cat((data3.aug_v0, data3.feat0), 1), data=data3, verbose=True)
                hm1, hm2 = hm1.detach(), hm2.detach()
            with T('encoder+prior'):
                pose_enc_0 = encoder_shape(data1.aug_v0, hm1, data=data1, feat=data1.feat0)
                pose_enc = encoder_shape(data1.aug_v1, hm1, data=data1, feat=data1.feat1)
                shape_enc = encoder_shape(data2.aug_v0, hm2, data=data2, feat=data2.feat0)
                with T('>get_transf'):
                    trans1 = get_transformation(hm1, region_score1, data1.batch, data1.aug_v0, data1.aug_v1)
                if(i_iter%60==0):
                    pred_v_carry=data2.aug_v1
                v4=encoder_shape(pred_v_carry, hm2, data=data2, feat=data2.feat0)
            
                if i_iter % 60 == 0:
                    v2 = torch.zeros_like(shape_enc)
                    window = []
                elif inp_iter == 0:
                    v2 = torch.zeros_like(shape_enc)          # baseline: no prior
                else:
                    window.append(v4.data)
                    v2 = aggregate_window(window, args.agg, device)
                    if len(window) >= inp_iter:               # keep at most `window` previous encodings
                        window.pop(0)
                
            with T('decode+mesh'):
                pred_disp = decoder(pose_enc-pose_enc_0, shape_enc, trans1.detach(),v2) 
                pred_v = handle2mesh(pred_disp, hd_pos2.detach(), region_score2, data2.batch, data2.aug_v0)
                pred_v_carry = pred_v.data
                pred_v_gtt = gt_trans_mesh(pred_disp, hm2, region_score2, data2.batch, data2.aug_v0, data2.v1)
            with T('losses'):
                loss1 = criterion(data2.aug_v1, pred_v) + criterion(data2.aug_v1, pred_v_gtt)
                loss1 = loss1 * 0.5
                with T('>transf_loss'):
                    loss3 = transformation_loss(pred_disp, hm2, region_score2, data2.batch, data2.aug_v0, data2.aug_v1, criterion) #+ \
                with T('>arap'):
                    loss_arap = arap_loss(data2.tpl_edge_index, data2.aug_v0, pred_v) * 100 #+ \         
                with T('>fps'):
                    fps_points0 = torch_geometric_fps(data2.aug_v0, data2.batch, num_handle*4, data2.ptr_list)
                loss_fps = distChamfer(fps_points0, hd_pos2, norm='L2')# + distChamfer(fps_points2, hd_pos3, norm='L2')
                with T('>sw_loss'):
                    loss_sw = sw_loss(region_score2, data2.weights, data2.batch) #+ \      
                loss_sw = loss_sw * 0.03
                loss_sw_sim = kl_div((region_score2+1e-6).log(), region_score2_noaug+1e-6)
                loss = loss1 + loss3 + loss_sw + loss_sw_sim + loss_fps + loss_arap
            with T('nan_check'):
                if torch.isinf(loss) or torch.isnan(loss):
                    print(f"Iter {i_iter}, invalid loss")   
                    continue
            with T('backward'):
                loss.backward()
            with T('optimizer'):
                optimizer.step()
            with T('log'):
                loss_m.add_loss(loss1=loss1, loss_sw=loss_sw, loss_sw_sim=loss_sw_sim,
                                loss_fps=loss_fps, loss_arap=loss_arap)
                if i_iter % 60 == 0:
                    loss_m.print_latest()
            # shape_enc1 = encoder_shape(pred_v, hm2, data=data2, feat=data2.feat0)
            # if((i_iter%60)==0):   # append features for first window-1 frames
            #     window.append(shape_enc1)
            # else:                       # replace the items in window
            #     window[0]=shape_enc1
            # print(len(window),window[0].shape)
            if i_iter % 60 == 0 or (args.val_every > 0 and (i_iter + 1) % args.val_every == 0):
                for name in loss_m.loss_names:
                    loss_m.writer.add_scalar(name, loss_m.accum_loss[name] / loss_m.iter, loss_m.epoch)
                    wandb_log[name] = loss_m.accum_loss[name] / loss_m.iter
            checkpoint = {"encoder_shape": encoder_shape.state_dict(),
                        "decoder": decoder.state_dict(),
                        "predictor": predictor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        'torch_rnd': torch.get_rng_state(), 'numpy_rnd': np.random.get_state(), 'epoch': i_epoch,
                        'window': args.window, 'agg': args.agg}      # read by demo.py to rebuild the prior
            epoch_done = (i_iter == n_iter - 1) or (bool(args.max_iters) and (i_iter + 1) >= args.max_iters)
            if (i_iter + 1) % args.ckpt_every == 0 or epoch_done:
                checkpoint['mid_epoch'] = not epoch_done
                checkpoint['iter'] = i_iter + 1
                torch.save(checkpoint, os.path.join(exp_dir, 'latest.pth'))
            if i_iter % 12000 == 0:   # snapshot of the start of every epoch
                torch.save(checkpoint, os.path.join(exp_dir, 'iter_{}.pth'.format(i_epoch)))
            stop_epoch = bool(args.max_iters) and (i_iter + 1) >= args.max_iters
            run_val = args.val_every > 0 and (i_iter + 1) % args.val_every == 0 and i_iter != 0
            if not run_val:
                if stop_epoch:
                    break
                continue

            total_iters = args.max_iters or n_iter
            final_val = i_epoch == args.epochs - 1 and (i_iter + 1) + args.val_every > total_iters   # no further validation will follow
            render_args = {"trans": (1.5, 0, 0), "center": False}
            save_dir = os.path.join(exp_dir, 'val_'+str(i_epoch)+"_"+str(i_iter))
            os.makedirs(save_dir, exist_ok=True)
            with torch.no_grad():
                encoder_shape.eval()
                decoder.eval()
                predictor.eval()
        
                print("executing validation........................")
                n_frames = n_nonfinite = 0
                mixamo_loss_m.update_epoch(i_epoch)    # fresh accumulators: the old code averaged over *all* validations of the epoch
                for i_iter1, (data1, data2) in enumerate(test_mixamo_loader):
                    data1.ptr_list, data2.ptr_list = data1.ptr.tolist(), data2.ptr.tolist()
                    data1.to(device)
                    data2.to(device)
                    hm0, hd0, _, region_score0 = predictor(torch.cat((data2.v0, data2.feat0), 1),
                                                        data=data2, verbose=True)
                    hm1, hd1, _, region_score1 = predictor(torch.cat((data1.v0, data1.feat0), 1),
                                                        data=data1, verbose=True)
                    trans1 = get_transformation(hm1, region_score1, data1.batch, data1.v0, data1.v1)
                    pose_enc_0 = encoder_shape(data1.v0, hm1, data=data1, feat=data1.feat0)
                    pose_enc = encoder_shape(data1.v1, hm1, data=data1, feat=data1.feat1)
                    shape_enc = encoder_shape(data2.v0, hm0, data=data2, feat=data2.feat0)
                    if(i_iter1%60==0):
                        pred_v2=data2.v1
                    v14 = encoder_shape(pred_v2, hm0, data=data2, feat=data2.feat0)
                    if i_iter1 % 60 == 0:
                        window1 = []
                        v12 = torch.zeros_like(shape_enc)
                    elif inp_iter == 0:
                        v12 = torch.zeros_like(shape_enc)
                    else:
                        window1.append(v14.data)
                        v12 = aggregate_window(window1, args.agg, device)
                        if len(window1) >= inp_iter:
                            window1.pop(0)
                    pred_disp = decoder(pose_enc-pose_enc_0, shape_enc, trans1 ,v12) 
                    pred_v1 = handle2mesh(pred_disp, hd0, region_score0, data2.batch, data2.v0)
                    pred_v2=pred_v1.data
                    dist = torch.mean(torch.sqrt(torch.sum((pred_v1 - data2.v1)**2, -1)))
                    loss = criterion(pred_v1, data2.v1)
                    n_frames += 1
                    if torch.isfinite(dist) and torch.isfinite(loss):
                        mixamo_loss_m.add_loss(mixamo_dist=dist, mixamo_loss=loss)
                    else:                    # rollout diverged: do not let one NaN poison the average, but report it
                        n_nonfinite += 1
                   
                    
                    
                    # visualization
                    if args.save_ply == 'all' or (args.save_ply == 'last' and final_val):
                        hd0 = hd0.detach().cpu().numpy()[0]
                        v, f, vc = visualize_handle(data2.v1.detach().cpu().numpy(), data2.triangle[0][0],
                                                    save_path=os.path.join(save_dir, f'mixamo_{i_iter1}_input.ply'))
                        # cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter1}_input.jpg'),
                        #             renderer(v, f, vc, **render_args))

                        Mesh(v=data1.v1.detach().cpu().numpy(), f=data1.triangle[0][0]).write_ply(
                            os.path.join(save_dir, f'mixamo_{i_iter1}_pose.ply'))
                        # cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter1}_pose.jpg'),
                        #             renderer(data1.v1.cpu().numpy(), data1.triangle[0][0], **render_args))

                        v, f, vc = visualize_handle(pred_v1.detach().cpu().numpy(), data2.triangle[0][0],
                                                    save_path=os.path.join(save_dir, f'mixamo_{i_iter1}_pred.ply'))
                        # cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter1}_pred.jpg'),
                        #             renderer(v, f, vc, **render_args))

                        # cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter1}_seg.jpg'),
                        #             renderer(*visualize_part(data2.v0, data2.triangle[0][0], hd0, region_score0,
                        #                                     save_path=os.path.join(save_dir, f'mixamo_{i_iter1}_seg.ply'))))
                    # f11=data1.triangle[0][0]
                    # f21=data2.triangle[0][0]
                    
                mixamo_loss_m.epoch_summary()
                v_dist = mixamo_loss_m.accum_loss['mixamo_dist'] / max(mixamo_loss_m.iter, 1) if mixamo_loss_m.iter else float('nan')
                v_loss = mixamo_loss_m.accum_loss['mixamo_loss'] / max(mixamo_loss_m.iter, 1) if mixamo_loss_m.iter else float('nan')
                g_step = i_epoch * n_iter + i_iter + 1
                print(f"VALIDATION_DONE iter {g_step} frames {n_frames} nonfinite {n_nonfinite} dist {v_dist:.6f} loss {v_loss:.6f}", flush=True)
                wandb_log['val_nonfinite_frames'] = n_nonfinite
                for k_, v_ in (('val/mixamo_dist', v_dist), ('val/mixamo_loss', v_loss), ('val/nonfinite_frames', n_nonfinite)):
                    loss_m.writer.add_scalar(k_, v_, g_step)
                wandb_log['mixamo_dist'], wandb_log['mixamo_loss'] = v_dist, v_loss

                wandb.log(wandb_log)
            if stop_epoch:
                break


if __name__ == '__main__':
    import signal
    try:
        main()
    except RuntimeError as e:
        print(e)
        os.kill(os.getpid(), signal.SIGINT)