#!/bin/bash
# Budgeted version of the report's experiments for a single 4 GB GPU:
# 6 runs (weighted window 1 == plain window 1, so it is skipped), ITERS training iterations each,
# validation every VAL iterations.  Finished runs are skipped and interrupted runs resume from their last checkpoint,
# so the script can simply be started again after any interruption.
#
# All outputs (checkpoints, tensorboard, logs) are written to RAM (/dev/shm) and mirrored to disk in the background:
# on this machine the btrfs root stalls for up to a minute at random, which used to freeze training whenever it wrote a file.
# usage: ./run_overnight.sh [ITERS=60000] [VAL=6000]
cd "$(dirname "$0")"; ROOT=$(pwd); source env.sh
ITERS=${1:-60000}; VAL=${2:-6000}
SHM=${SHM:-/dev/shm/sfmt_runs}; DISK=${DISK:-$ROOT/../runs}      # override both for dry runs
mkdir -p "$SHM/logs" "$DISK"
rsync -a --ignore-existing "$DISK/" "$SHM/"            # restore after a reboot / earlier interruption
export SFMT_LOG_DIR="$SHM" WANDB_DIR="$SHM"
( while true; do nice -n 10 rsync -a "$SHM/" "$DISK/"; sleep 120; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; rsync -a "$SHM/" "$DISK/"' EXIT

run() {   # name window agg ; resumes from latest.pth after a crash and retries (e.g. transient GPU faults)
  name=$1; w=$2; agg=$3; extra=$4                      # extra: additional train.py flags, e.g. "--seed 1"
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi   # ONLY="baseline avg_w5" restricts the set
  if [ -f "$SHM/$name/DONE" ]; then echo "skip $name"; return; fi
  for attempt in 1 2 3 4 5 6; do
    echo "=== $name attempt $attempt $(date)"
    python -u train.py --window $w --agg $agg --epochs 1 --max_iters $ITERS --val_every $VAL \
           --wandb offline --save_ply last --exp_name $name $extra >> "$SHM/logs/$name.log" 2>&1
    rc=$?
    expected=$(( ITERS / VAL ))                          # validation passes a complete run performs (ITERS should be a multiple of VAL)
    got=$(grep -c "^VALIDATION_DONE" "$SHM/logs/$name.log")
    if [ $rc -eq 0 ] && [ "$got" -ge "$expected" ]; then touch "$SHM/$name/DONE"; echo "=== $name finished $(date)"; return; fi
    echo "=== $name failed (rc=$rc) $(date); retrying in 120s"; sleep 120
  done
  echo "=== $name GAVE UP"
}
run baseline      0 avg
run avg_w5        5 avg
run weighted_w5   5 weighted_avg
run avg_w3        3 avg
run weighted_w3   3 weighted_avg
run avg_w1        1 avg
# replicates with another seed (model init + data schedule) to measure seed-to-seed noise
run baseline_s1   0 avg "--seed 1"
run avg_w5_s1     5 avg "--seed 1"
echo ALL_RUNS_DONE $(date)
