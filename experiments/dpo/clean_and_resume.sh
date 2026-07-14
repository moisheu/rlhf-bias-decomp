#!/usr/bin/env bash
# Remove everything downstream of the EOS fix that might have been stale-
# reused (relabel outputs, DPO checkpoints, generations), then resume the
# pipeline. Deliberately keeps: RM checkpoints, Phase 3 tags/weights, the
# freshly-fixed sft_base, and the raw dpo_pairs.json/sft_texts.json source
# data -- none of those are affected by the EOS fix, no need to redo them.
#
# Usage (inside tmux, from the repo root):
#   bash experiments/dpo/clean_and_resume.sh
set -eu
cd /workspace/rlhf-bias-decomp

echo "===== Removing stale relabel outputs (need the EOS-append fix) ====="
rm -f results/phase4/dpo_labeled_human.json results/phase4/dpo_labeled_raw.json results/phase4/dpo_labeled_reweight.json
rm -f results/phase4/prefmask_human.json results/phase4/prefmask_raw.json results/phase4/prefmask_reweight.json
rm -f results/phase4/disagree_reweight_vs_raw.json

echo "===== Removing stale DPO checkpoints (must retrain on fixed SFT base + fixed labels) ====="
rm -rf results/dpo_raw_seed42 results/dpo_raw_seed0 results/dpo_human_seed42 results/dpo_human_seed0 results/dpo_reweight_seed42 results/dpo_reweight_seed0

echo "===== Removing stale generations ====="
rm -f results/phase4/gen_sft.json results/phase4/gen_raw_seed42.json results/phase4/gen_raw_seed0.json results/phase4/gen_human_seed42.json results/phase4/gen_human_seed0.json results/phase4/gen_reweight_seed42.json results/phase4/gen_reweight_seed0.json

echo "===== Kept (unaffected by the EOS fix): RM checkpoints, Phase 3 tags/weights, sft_base, raw source data ====="
ls results/reward_model_mixed_seed42/model.safetensors results/reward_model_reweight_seed0/model.safetensors results/phase4/sft_base/model.safetensors 2>/dev/null && echo "  all present, good"

echo "===== Resuming pipeline ====="
bash experiments/dpo/run_phase4_full.sh 2>&1 | tee -a results/phase4_full.log

echo "===== DONE ====="
