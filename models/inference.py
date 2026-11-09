"""Inference helpers shared by demo.py and demo_smpl.py.

Loads a checkpoint (original SFPT or one trained with train.py, with or without the window prior) and transfers
poses frame by frame.  For a checkpoint with the prior, each target character keeps a `TargetState` that holds the
encodings of its last `window` predicted frames, exactly like the validation loop of train.py:
  * first frame of a sequence: prior = zeros (no previous prediction yet)
  * later frames: encode the previous *predicted* mesh of this target, push it into the window, aggregate
"""
import torch

from models.networks import PerPartEncoderTpl, PerPartDecoder, HandlePredictorSWTpl
from models.ops import handle2mesh, get_transformation, arap_smooth
from models.prior import aggregate_window

FEAT = 128
INPUT_DIM = 6                       # position + normal
NUM_HANDLE = 40
DECODER_IN_ORIGINAL = 2 * FEAT + 7  # original SFPT decoder: pose code, shape code, handle transformation
DECODER_IN_PRIOR = 3 * FEAT + 7     # + the prior input (zeros for the baseline, window aggregate otherwise)


class Models:
    def __init__(self, encoder_shape, encoder_pose, predictor, decoder, window, agg, with_prior, device):
        self.encoder_shape, self.encoder_pose, self.predictor, self.decoder = encoder_shape, encoder_pose, predictor, decoder
        self.window, self.agg, self.with_prior, self.device = window, agg, with_prior, device

    def describe(self):
        if not self.with_prior:
            return "original SFPT decoder (no prior input)"
        if self.window == 0:
            return "decoder with prior input, window 0 (baseline, prior = zeros)"
        return f"decoder with prior input, window {self.window}, aggregation {self.agg}"


def load_models(ckpt_path, device, window=None, agg=None):
    """window / agg: values used in training.  train.py records them in the checkpoint; for checkpoints that do not
    (older runs) they must be given.  Checkpoints of the original SFPT (no prior input) need neither."""
    device = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    decoder_in = ckpt["decoder"]["net.0.weight"].shape[1]
    if decoder_in == DECODER_IN_ORIGINAL:
        with_prior = False
        if window:
            raise ValueError(f"{ckpt_path} is an original SFPT checkpoint (decoder without prior input); --window {window} does not apply")
        window, agg = 0, "avg"
    elif decoder_in == DECODER_IN_PRIOR:
        with_prior = True
        window = ckpt.get("window") if window is None else window
        agg = ckpt.get("agg") if agg is None else agg
        if window is None:
            raise ValueError("this checkpoint has a prior input but does not record the window size it was trained with: "
                             "pass --window N (0 = baseline) and --agg avg|weighted_avg, as used in training")
        if window > 0 and agg is None:
            raise ValueError("pass --agg avg|weighted_avg (the aggregation used in training)")
        agg = agg or "avg"
    else:
        raise ValueError(f"unexpected decoder input size {decoder_in} in {ckpt_path}")

    encoder_shape = PerPartEncoderTpl(INPUT_DIM, FEAT)
    encoder_pose = PerPartEncoderTpl(INPUT_DIM, FEAT)
    predictor = HandlePredictorSWTpl(INPUT_DIM, NUM_HANDLE)
    decoder = PerPartDecoder(decoder_in)
    encoder_shape.load_state_dict(ckpt["encoder_shape"])
    # train.py uses one encoder for shape and pose (no 'encoder_pose' key); the original SFPT checkpoints have both
    encoder_pose.load_state_dict(ckpt.get("encoder_pose", ckpt["encoder_shape"]))
    decoder.load_state_dict(ckpt["decoder"])
    predictor.load_state_dict(ckpt["predictor"])
    nets = [encoder_shape, encoder_pose, predictor, decoder]
    for m in nets:
        m.to(device).eval()
    return Models(*nets, window, agg, with_prior, device)


class TargetState:
    """Per-target-character memory of the already predicted frames (keep one per target, start a new one per sequence)."""
    def __init__(self):
        self.prev = None      # previous predicted mesh (normalised coordinates), None before the first frame
        self.window = []      # encodings of the last predicted frames


@torch.no_grad()
def transfer_frame(m, src, dst, state, smooth=False):
    """Pose of `src` (rest v0, posed v1) -> character `dst` (rest v0).  Returns the predicted mesh in the target's
    normalised coordinates ((v - dst.center) / dst.scale).  `state` is updated in place."""
    hm0, hd0_mean, _, score0 = m.predictor(torch.cat((dst.v0, dst.feat0), 1), data=dst, verbose=True)
    hm1, _, _, score1 = m.predictor(torch.cat((src.v0, src.feat0), 1), data=src, verbose=True)
    trans1 = get_transformation(hm1, score1, src.batch, src.v0, src.v1)
    ap1 = m.encoder_pose(src.v1, hm1, data=src, feat=src.feat1)
    as0 = m.encoder_shape(src.v0, hm1, data=src, feat=src.feat0)
    rs0 = m.encoder_shape(dst.v0, hm0, data=dst, feat=dst.feat0)
    feats = [ap1 - as0, rs0, trans1]
    if m.with_prior:
        if state.prev is None or m.window == 0:
            prior = torch.zeros_like(rs0)
            state.window = []
        else:
            state.window.append(m.encoder_shape(state.prev, hm0, data=dst, feat=dst.feat0).data)
            prior = aggregate_window(state.window, m.agg, m.device)
            if len(state.window) >= m.window:
                state.window.pop(0)
        feats.append(prior)
    pred_disp = m.decoder(*feats)
    pred_v = handle2mesh(pred_disp, hd0_mean, score0, dst.batch, dst.v0)
    state.prev = pred_v.data                      # the prior sees the raw prediction, as in training
    if smooth:
        pred_disp = arap_smooth(pred_disp, hd0_mean, score0, dst.batch, dst.v0, dst.tpl_edge_index, 0.5)
        pred_v = handle2mesh(pred_disp, hd0_mean, score0, dst.batch, dst.v0)
    return pred_v
