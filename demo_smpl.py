"""Transfer a SMPL pose to another SMPL body shape (needs the SMPL-H model files, see global_var.SMPLH_PATH).

  python demo_smpl.py --ckpt demo/ckpt.pth
  python demo_smpl.py --ckpt runs/avg_w5/latest.pth --window 5 --agg avg

Writes the source (rest + one pose) and the target body to --src / --dst, then transfers like demo.py.
Output: demo/results_smpl/<target>_<frame>.obj
"""
import argparse
import os

import numpy as np

import global_var
from demo_common import add_args, transfer_all
from models.smpl import SMPL2Mesh
from utils.o3d_wrapper import Mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="./demo/src_data_smpl", help="where the generated source meshes are written")
    parser.add_argument("--dst", default="./demo/dst_data_smpl", help="where the generated target body is written")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random body shapes (default: random)")
    add_args(parser, out="./demo/results_smpl", smooth=False)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    os.makedirs(args.src, exist_ok=True)
    os.makedirs(os.path.join(args.dst, 'obj_remesh'), exist_ok=True)

    smpl = SMPL2Mesh(global_var.SMPLH_PATH)
    pose0 = np.zeros([1, 156])
    pose1 = np.zeros([1, 156])
    pose1[0, 18*3+1] = -np.pi/2
    shape1 = np.random.normal(loc=0, scale=2, size=[1, 16])
    shape2 = np.random.normal(loc=0, scale=2, size=[1, 16])
    src0 = smpl(pose0, shape1, [1])
    src1 = smpl(pose1, shape1, [1])
    dst = smpl(pose0, shape2, [1])
    Mesh(v=src0[0].numpy(), f=smpl.f).write_obj(os.path.join(args.src, 'rest.obj'))
    Mesh(v=src1[0].numpy(), f=smpl.f).write_obj(os.path.join(args.src, '1.obj'))
    Mesh(v=dst[0].numpy(), f=smpl.f).write_obj(os.path.join(args.dst, 'obj_remesh', 'dst1.obj'))

    transfer_all(args, args.src, args.dst)
