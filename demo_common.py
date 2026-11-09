"""Shared by demo.py and demo_smpl.py: load a checkpoint and transfer every source frame to every target character."""
import argparse
import os

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_utils.custom_loader import CustomDataset, CustomMotionDataset
from models.inference import load_models, TargetState, transfer_frame
from utils.o3d_wrapper import Mesh


def add_args(parser, out, smooth):
    parser.add_argument("--ckpt", default="./demo/ckpt.pth", help="checkpoint: the original SFPT one, or a latest.pth written by train.py")
    parser.add_argument("--out", default=out, help="output directory for the transferred meshes (<target>_<frame>.obj)")
    parser.add_argument("--window", type=int, default=None,
                        help="window size the checkpoint was trained with (0 = baseline). Not needed for the original SFPT checkpoint or "
                             "for checkpoints written by the current train.py, which record it")
    parser.add_argument("--agg", choices=["avg", "weighted_avg"], default=None,
                        help="aggregation the checkpoint was trained with (same remark as --window)")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=smooth, help="ARAP smoothing of the output")
    parser.add_argument("--device", default="cuda:0")


def transfer_all(args, src_dir, dst_dir):
    """src_dir: rest.obj + <frame>.obj files, played in numeric order as ONE continuous motion (the window prior carries
    information from frame to frame).  dst_dir: obj_remesh/<character>.obj, the target characters in rest pose."""
    os.makedirs(args.out, exist_ok=True)
    try:
        models = load_models(args.ckpt, args.device, args.window, args.agg)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    print("model:", models.describe())

    src_loader = DataLoader(CustomMotionDataset(src_dir), batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    dst_loader = DataLoader(CustomDataset(dst_dir), batch_size=1, shuffle=False, pin_memory=False, drop_last=False)
    dsts = [d.to(models.device) for d in dst_loader]
    states = [TargetState() for _ in dsts]             # one memory of predicted frames per target character
    print(f"{len(src_loader)} source frames x {len(dsts)} target characters")
    for src in tqdm(src_loader):
        src.to(models.device)
        for dst, state in zip(dsts, states):
            pred_v = transfer_frame(models, src, dst, state, smooth=args.smooth)
            pred_v = (pred_v * dst.scale + dst.center).cpu().numpy()
            Mesh(v=pred_v, f=dst.triangle[0][0]).write_obj(os.path.join(args.out, f'{dst.name[0]}_{src.name[0]}.obj'))
    print("wrote", args.out)
