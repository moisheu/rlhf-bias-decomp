#!/usr/bin/env bash
# Re-run generation ONLY (no retraining) at a higher max_new_tokens, then
# re-print the decision table. Fixes the truncation issue where every policy's
# median generation hit the old 256-token cap, compressing the exact length
# differences Phase 4 is trying to measure (spec: "if >20% truncation, raise
# the cap and regenerate -- eval-only, cheap"). Unconditionally overwrites the
# existing gen_*.json files (no skip-if-exists), unlike run_phase4_full.sh.
#
# Usage on the pod:
#   curl -H "Authorization: token <YOUR_GITHUB_TOKEN>" -o regen.sh \
#     https://raw.githubusercontent.com/moisheu/rlhf-bias-decomp/phase3-decomposition/experiments/dpo/regenerate_full_length.sh
#   bash regen.sh
set -eu
cd /rlhf-bias-decomp

RAW_RM=results/reward_model_mixed_seed42
RW_RM=results/reward_model_reweight_seed0
NEW_TOKENS=512

gen() {
  local policy_dir=$1 label=$2
  echo "===== generating: $label ====="
  python -u -m experiments.dpo.generate \
    --policy-dir "$policy_dir" --label "$label" \
    --raw-rm "$RAW_RM" --reweight-rm "$RW_RM" \
    --max-new-tokens "$NEW_TOKENS"
}

gen results/phase4/sft_base sft
gen results/dpo_raw_seed42 raw_seed42
gen results/dpo_raw_seed0 raw_seed0
gen results/dpo_human_seed42 human_seed42
gen results/dpo_human_seed0 human_seed0
gen results/dpo_reweight_seed42 reweight_seed42
gen results/dpo_reweight_seed0 reweight_seed0

echo "===== decision table ====="
python -m experiments.dpo.summarize_phase4
