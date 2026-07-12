#!/usr/bin/env bash
# Phase 4 full pipeline — for a persistent SSH GPU box (RunPod, Lambda Labs, etc.),
# NOT Colab. Meant to run inside `tmux` so it survives disconnects: everything lives
# on local disk, nothing depends on a browser session or Drive.
#
# Usage (on the GPU box, inside a tmux session):
#   git clone --branch phase3-decomposition https://<TOKEN>@github.com/moisheu/rlhf-bias-decomp.git
#   cd rlhf-bias-decomp
#   pip install -q 'transformers==5.9.0' 'trl==1.5.0' 'datasets>=2.14.0' 'accelerate>=0.27.0' scipy numpy
#   bash experiments/dpo/run_phase4_full.sh 2>&1 | tee results/phase4_full.log
#
# Resumable: every stage is guarded by a "does the output already exist" check, so
# re-running this script after a crash/reboot picks up where it left off — same
# pattern as experiments/decomposition/run_queue.sh, just without the Drive dance
# since this box's disk IS the persistent store.
set -eu
cd "$(dirname "$0")/../.."

RAW_RM=results/reward_model_mixed_seed42
RW_RM=results/reward_model_reweight_seed0
SFT_DIR=results/phase4/sft_base

has_weights() { [ -f "$1/model.safetensors" ] || [ -f "$1/pytorch_model.bin" ]; }

echo "===== [1/7] Phase 3 tags + weights (deterministic, cheap) ====="
[ -f results/mixed_pool_subset_tags_seed42.json ] || python -m experiments.decomposition.build_subset_tags
[ -f results/reweight_weights_seed42pool.json ]   || python -m experiments.decomposition.compute_weights

echo "===== [2/7] Relabeler RMs ====="
if ! has_weights "$RAW_RM"; then
  DATA_MODE=mixed TRAIN_SEED=42 python -m src.train_reward_model
fi
if ! has_weights "$RW_RM"; then
  METHOD=reweight TRAIN_SEED=0 TRAIN_BATCH=16 python -m experiments.decomposition.train_phase3
fi
python -m experiments.decomposition.eval_length_correlation --model-dir "$RAW_RM" --label raw_relabeler --out results/phase4_relabeler_rms.json
python -m experiments.decomposition.eval_length_correlation --model-dir "$RW_RM"  --label reweight_relabeler --out results/phase4_relabeler_rms.json
cat results/phase4_relabeler_rms.json

echo "===== [3/7] Phase 4 data + relabeling (Day 1) ====="
[ -f results/phase4/dpo_pairs.json ] || python -m experiments.dpo.build_phase4_data
[ -f results/phase4/dpo_labeled_human.json ]    || python -m experiments.dpo.relabel --labeler human
[ -f results/phase4/dpo_labeled_raw.json ]      || python -m experiments.dpo.relabel --labeler raw --model-dir "$RAW_RM"
[ -f results/phase4/dpo_labeled_reweight.json ] || python -m experiments.dpo.relabel --labeler reweight --model-dir "$RW_RM"

echo "===== [4/7] Shared SFT base ====="
has_weights "$SFT_DIR" || python -m experiments.dpo.train_sft

echo "===== [5/7] DPO runs (6 arms, kill-switch: 1 LR halving then stop) ====="
run_dpo() {
  local labeler=$1 seed=$2
  for lr in 5e-6 2.5e-6; do
    LABELER=$labeler TRAIN_SEED=$seed DPO_LR=$lr TRAIN_BATCH=8 \
      python -u -m experiments.dpo.train_dpo && return 0
    rc=$?
    if [ $rc -eq 2 ]; then
      echo "  ${labeler}_seed${seed} unstable at lr=$lr — trying next"
      continue
    fi
    echo "  ${labeler}_seed${seed} FAILED rc=$rc (not instability)" >&2
    return 1
  done
  return 2  # exhausted retries = kill-switch
}
for arm in "raw 42" "human 42" "reweight 42" "raw 0" "human 0" "reweight 0"; do
  set -- $arm
  labeler=$1; seed=$2
  outdir="results/dpo_${labeler}_seed${seed}"
  if has_weights "$outdir"; then
    echo "SKIP dpo_${labeler}_seed${seed} (exists)"; continue
  fi
  echo "--- DPO ${labeler}_seed${seed} ---"
  if ! run_dpo "$labeler" "$seed"; then
    echo "KILL-SWITCH: ${labeler}_seed${seed} unstable after one LR halving. Stopping (do not tune)." >&2
    exit 1
  fi
done

echo "===== [6/7] Generation (7 policies, cross-scored) ====="
gen() {
  local policy_dir=$1 label=$2
  [ -f "results/phase4/gen_${label}.json" ] && { echo "SKIP gen_${label}"; return; }
  python -u -m experiments.dpo.generate --policy-dir "$policy_dir" --label "$label" \
    --raw-rm "$RAW_RM" --reweight-rm "$RW_RM"
}
gen "$SFT_DIR" sft
for arm in "raw 42" "human 42" "reweight 42" "raw 0" "human 0" "reweight 0"; do
  set -- $arm
  gen "results/dpo_$1_seed$2" "$1_seed$2"
done

echo "===== [7/7] Decision table ====="
python -m experiments.dpo.summarize_phase4

echo "===== PHASE 4 COMPLETE ====="
