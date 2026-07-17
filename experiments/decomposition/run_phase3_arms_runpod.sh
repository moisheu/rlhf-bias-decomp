#!/usr/bin/env bash
# One-shot: clone the repo (or update it if already present), install deps,
# retrain all 9 Phase 3 arms (symloss/reweight/combined x seeds 42/0/1),
# eval length correlations, reconcile against the Writing Reference numbers,
# commit + push the results JSON, then stop the pod. For a fresh RunPod GPU
# pod with a Network Volume mounted (so checkpoints + git history survive
# the pod going away).
#
# Model checkpoints are NEVER added to git -- they live only under
# REPO_DIR/results/reward_model_*_seed*/ on the network volume. Only the
# small length-correlation JSON (+ a short reconciliation report) get
# committed and pushed.
#
# Usage (inside tmux, from a fresh pod shell):
#   export GITHUB_TOKEN=your_github_token       # repo scope, private repo
#   export HF_TOKEN=your_huggingface_token       # read scope, dataset loads
#   curl -H "Authorization: token $GITHUB_TOKEN" -o run_phase3_arms_runpod.sh \
#       https://raw.githubusercontent.com/moisheu/rlhf-bias-decomp/phase3-decomposition/experiments/decomposition/run_phase3_arms_runpod.sh
#   bash run_phase3_arms_runpod.sh
#
# Optional overrides:
#   REPO_DIR=/workspace/persist/rlhf-bias-decomp   (default; network volume)
#   AUTO_SHUTDOWN=1                                 (default; 0 to disable)
set -eu

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN=<your token> before running this script}"
: "${HF_TOKEN:?Set HF_TOKEN=<your token> before running this script}"
REPO_DIR="${REPO_DIR:-/workspace/persist/rlhf-bias-decomp}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
BRANCH=phase3-decomposition
REPO_URL="https://${GITHUB_TOKEN}@github.com/moisheu/rlhf-bias-decomp.git"
VENV="$HOME/.venvs/rlhf-bias"

echo "===== [1/9] Clone / update repo ====="
mkdir -p "$(dirname "$REPO_DIR")"
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  # Re-inject the token: a prior run may have scrubbed it (see end of script),
  # and private-repo fetch/pull needs it every time, not just on first clone.
  git remote set-url origin "$REPO_URL"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
echo "On branch: $(git branch --show-current)   Latest commit: $(git log --oneline -1)"

echo "===== [2/9] Environment setup ====="
python3 --version
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q 'transformers==5.9.0' 'trl==1.5.0' 'datasets==4.8.5' 'accelerate==1.13.0' 'scipy==1.17.1' 'numpy==2.4.6'
python3 -c "
import torch, sys
ok = torch.cuda.is_available()
print('torch', torch.__version__, '| cuda:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU')
sys.exit(0 if ok else 1)
"

echo "===== [3/9] HuggingFace auth (via HF_TOKEN env var, library-level check) ====="
# Not shelling out to huggingface-cli/hf -- that CLI's name/flags have already
# churned once (cli -> hf) and datasets/transformers read HF_TOKEN from the
# environment directly regardless, so a library-level whoami() is both the
# real validation and immune to future CLI renames.
python3 -c "
import os
from huggingface_hub import whoami
info = whoami(token=os.environ['HF_TOKEN'])
print('HF auth OK, token valid for:', info.get('name', info))
"

echo "===== [4/9] Sanity check: 100-row hh-rlhf mixed pool load ====="
python3 -c "
from src.data_utils import load_hh_rlhf_mixed
ds = load_hh_rlhf_mixed(n=100, seed=42)
assert len(ds) == 100, f'expected 100 rows, got {len(ds)}'
assert ds.column_names == ['chosen', 'rejected'], ds.column_names
print(len(ds), ds.column_names)
print(ds['chosen'][0][:200])
"
echo "Sanity check passed."

echo "===== [5/9] Confirm subset tags + reweighting weights exist ====="
if [ ! -f results/mixed_pool_subset_tags_seed42.json ]; then
  echo "MISSING mixed_pool_subset_tags_seed42.json -- rebuilding..."
  python -m experiments.decomposition.build_subset_tags
fi
if [ ! -f results/reweight_weights_seed42pool.json ]; then
  echo "MISSING reweight_weights_seed42pool.json -- rebuilding..."
  python -m experiments.decomposition.compute_weights
fi
ls -la results/mixed_pool_subset_tags_seed42.json results/reweight_weights_seed42pool.json

echo "===== [6/9] The nine training runs (TRAIN_BATCH=16, matches Colab) ====="
mkdir -p results/phase3_logs
has_weights() { [ -f "$1/model.safetensors" ] || [ -f "$1/pytorch_model.bin" ]; }
ARMS=(
  "symloss 42" "reweight 42" "combined 42"
  "symloss 0"  "reweight 0"  "combined 0"
  "symloss 1"  "reweight 1"  "combined 1"
)
for arm in "${ARMS[@]}"; do
  set -- $arm
  METHOD=$1; SEED=$2
  tag="${METHOD}_seed${SEED}"
  outdir="results/reward_model_${tag}"
  if has_weights "$outdir"; then
    echo "[$(date +%H:%M:%S)] SKIP $tag (already trained)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] TRAIN start $tag"
  rm -rf "$outdir"
  if TRAIN_BATCH=16 METHOD=$METHOD TRAIN_SEED=$SEED python -u -m experiments.decomposition.train_phase3 \
      > "results/phase3_logs/runpod_${tag}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] TRAIN done $tag"
  else
    echo "[$(date +%H:%M:%S)] TRAIN FAILED $tag -- see results/phase3_logs/runpod_${tag}.log -- continuing"
  fi
done
echo "All 9 arms attempted."

echo "===== [7/9] Post-training length-correlation eval ====="
for arm in "${ARMS[@]}"; do
  set -- $arm
  tag="${1}_seed${2}"
  outdir="results/reward_model_${tag}"
  if ! has_weights "$outdir"; then
    echo "SKIP eval $tag (no checkpoint -- training failed or never ran)"
    continue
  fi
  python -u -m experiments.decomposition.eval_length_correlation \
    --model-dir "$outdir" --label "$tag" --out results/phase3_arms_length_corr.json
done
cat results/phase3_arms_length_corr.json

echo "===== [8/9] Reconcile against the Writing Reference ====="
python3 - << 'PY' | tee results/phase3_arms_reconciliation.txt
import json, os

DOC = {
    ("symloss", 42): 0.250, ("symloss", 0): 0.422, ("symloss", 1): 0.545,
    ("reweight", 42): 0.304, ("reweight", 0): 0.289, ("reweight", 1): 0.331,
    ("combined", 42): 0.247, ("combined", 0): 0.257, ("combined", 1): 0.494,
}
DOC_MEAN = {"symloss": 0.406, "reweight": 0.308, "combined": 0.333}

def flag(delta):
    if abs(delta) <= 0.02: return "OK (fp/RNG drift)"
    if abs(delta) > 0.05: return "*** REAL MISMATCH ***"
    return "moderate, below 0.05 line"

path = "results/phase3_arms_length_corr.json"
rows = {r["label"]: r for r in json.load(open(path))} if os.path.exists(path) else {}

print(f"{'label':20}{'doc r':>9}{'new r':>9}{'delta':>9}  flag")
by_method = {}
for (method, seed), doc_r in DOC.items():
    tag = f"{method}_seed{seed}"
    row = rows.get(tag)
    if row is None:
        print(f"{tag:20}{doc_r:9.3f}{'MISSING':>9}")
        continue
    new_r = row["r_pooled"]
    by_method.setdefault(method, []).append(new_r)
    d = new_r - doc_r
    print(f"{tag:20}{doc_r:9.3f}{new_r:9.3f}{d:+9.3f}  {flag(d)}")

print(f"\n{'method mean':20}{'doc mean':>9}{'new mean':>9}{'delta':>9}  flag")
for method, vals in by_method.items():
    new_mean = sum(vals) / len(vals)
    d = new_mean - DOC_MEAN[method]
    print(f"{method:20}{DOC_MEAN[method]:9.3f}{new_mean:9.3f}{d:+9.3f}  {flag(d)}")
PY

echo "===== [9/9] Commit + push (checkpoints excluded) ====="
git check-ignore -v results/reward_model_reweight_seed42/model.safetensors results/reward_model_symloss_seed0/model.safetensors || true
git add -f results/phase3_arms_length_corr.json results/phase3_arms_reconciliation.txt
git status --porcelain

PUSH_OK=0
if ! git diff --cached --quiet; then
  git commit -m "phase3: retrain arms on RunPod, persist length correlations (checkpoints stay on network volume, not git)"
  git push origin "$BRANCH"
  if [ -z "$(git log origin/"$BRANCH"..HEAD --oneline)" ]; then
    echo "Push confirmed: origin/$BRANCH is up to date with HEAD."
    PUSH_OK=1
  else
    echo "WARNING: push did not fully land -- origin/$BRANCH is behind HEAD. NOT auto-stopping the pod." >&2
  fi
else
  echo "Nothing new to commit (already up to date) -- treating as pushed."
  PUSH_OK=1
fi

# Scrub the token from the stored remote now that all git network ops for
# this run are done (re-injected at the top of [1/9] on the next invocation).
git remote set-url origin "https://github.com/moisheu/rlhf-bias-decomp.git"

echo "===== DONE. Reconciliation report: results/phase3_arms_reconciliation.txt ====="

if [ "$AUTO_SHUTDOWN" = "1" ] && [ "$PUSH_OK" = "1" ]; then
  echo ""
  echo "#####################################################################"
  echo "# AUTO_SHUTDOWN=1 and push confirmed -- stopping this pod in 60s.    #"
  echo "# Ctrl-C now to cancel (results are already safely pushed either way)."
  echo "#####################################################################"
  for i in 60 50 40 30 20 10; do
    echo "  stopping in ${i}s..."
    sleep 10
  done
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  else
    echo "runpodctl or RUNPOD_POD_ID not available -- STOP THE POD MANUALLY NOW via the RunPod web UI."
  fi
else
  echo ""
  echo "STOP THE POD NOW: runpodctl stop pod <POD_ID> (get <POD_ID> via 'runpodctl get pod'), or use the RunPod web UI."
fi
