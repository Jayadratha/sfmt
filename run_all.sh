#!/bin/bash
# Run every window experiment from the report: baseline (no prior) + window {1,3,5} x {avg, weighted_avg}.
# Usage: ./run_all.sh [extra train.py args, e.g. --epochs 10 --max_iters 200]
set -e
cd "$(dirname "$0")"
EXTRA="$@"
python train.py --window 0 $EXTRA
for agg in avg weighted_avg; do
  for w in 1 3 5; do
    python train.py --window $w --agg $agg $EXTRA
  done
done
