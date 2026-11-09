"""Transfer a source motion (obj frames) onto target characters (obj, rest pose).

  python demo.py --ckpt demo/ckpt.pth                                     # original SFPT checkpoint
  python demo.py --ckpt runs/avg_w5/latest.pth --window 5 --agg avg       # checkpoint trained here (see --window / --agg)

Input layout:   demo/src_data/rest.obj, 0.obj, 1.obj, ...      demo/dst_data/obj_remesh/<character>.obj
Output:         demo/results/<character>_<frame>.obj
"""
import argparse

from demo_common import add_args, transfer_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="./demo/src_data", help="source motion directory")
    parser.add_argument("--dst", default="./demo/dst_data", help="target characters directory")
    add_args(parser, out="./demo/results", smooth=True)
    args = parser.parse_args()
    transfer_all(args, args.src, args.dst)
