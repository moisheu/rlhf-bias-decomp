#!/usr/bin/env bash
# Phase 3 serial run orchestrator (single MPS device -> runs must be serial).
#
# Runs each arm as: train_phase3 -> eval_length_correlation (appending the
# Chunk 9 table to results/phase3_length_corr.json). Resumable: an arm with a
# .done marker is skipped, so re-launching continues where it left off.
#
# Arm order is first-seeds-first (symloss/reweight/combined at seed 42, then
# seed 0, then seed 1) so each arm's primary (seed-42) result lands earliest.
# Combined is CORE, not optional (per instructions), run at all three seeds.
#
# Run from repo root (inside the venv):
#   nohup bash experiments/decomposition/run_queue.sh \
#       > results/phase3_logs/queue.log 2>&1 &
set -u

LOGDIR=results/phase3_logs
STATUS=$LOGDIR/queue_status.txt
CORR_OUT=results/phase3_length_corr.json
mkdir -p "$LOGDIR"

# "METHOD SEED" — first-seeds first.
ARMS=(
  "symloss 42"
  "reweight 42"
  "combined 42"
  "symloss 0"
  "reweight 0"
  "combined 0"
  "symloss 1"
  "reweight 1"
  "combined 1"
)

echo "queue started $(date)" >> "$STATUS"

for arm in "${ARMS[@]}"; do
  set -- $arm
  METHOD=$1; SEED=$2
  tag="${METHOD}_seed${SEED}"
  marker="$LOGDIR/${tag}.done"
  outdir="results/reward_model_${tag}"

  if [ -f "$marker" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $tag (marker exists)" >> "$STATUS"
    continue
  fi

  echo "[$(date +%H:%M:%S)] TRAIN start $tag" >> "$STATUS"
  rm -rf "$outdir"
  METHOD=$METHOD TRAIN_SEED=$SEED python -u -m experiments.decomposition.train_phase3 \
      > "$LOGDIR/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] TRAIN FAILED $tag rc=$rc — stopping queue" >> "$STATUS"
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] TRAIN done $tag" >> "$STATUS"

  echo "[$(date +%H:%M:%S)] EVAL start $tag" >> "$STATUS"
  python -u -m experiments.decomposition.eval_length_correlation \
      --model-dir "$outdir" --label "$tag" --out "$CORR_OUT" \
      > "$LOGDIR/${tag}.eval.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] EVAL FAILED $tag rc=$rc — stopping queue" >> "$STATUS"
    exit 1
  fi
  touch "$marker"
  # surface the one-line result into the status file
  grep -h "\[$tag\]" "$LOGDIR/${tag}.eval.log" | tail -1 >> "$STATUS"
  echo "[$(date +%H:%M:%S)] EVAL done $tag" >> "$STATUS"
done

echo "queue finished $(date)" >> "$STATUS"
