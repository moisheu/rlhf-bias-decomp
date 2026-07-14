#!/usr/bin/env bash
# One-shot: clone the repo (or update it if already present), install deps,
# and run the ENTIRE Phase 4 pipeline end to end. For use on a fresh pod.
#
# Usage on the pod (inside tmux!):
#   export GITHUB_TOKEN=your_token_here
#   curl -H "Authorization: token $GITHUB_TOKEN" -o full_restart.sh https://raw.githubusercontent.com/moisheu/rlhf-bias-decomp/phase3-decomposition/experiments/dpo/full_restart.sh
#   bash full_restart.sh
set -eu

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN=<your token> before running this script}"
REPO_DIR="${REPO_DIR:-/workspace/rlhf-bias-decomp}"
BRANCH=phase3-decomposition
REPO_URL="https://${GITHUB_TOKEN}@github.com/moisheu/rlhf-bias-decomp.git"

echo "===== Clone / update repo ====="
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi
echo "On branch: $(git branch --show-current)"
echo "Latest commit: $(git log --oneline -1)"

echo "===== Install pinned dependencies ====="
pip install -q transformers==5.9.0 trl==1.5.0 datasets accelerate scipy numpy

echo "===== Running full Phase 4 pipeline ====="
bash experiments/dpo/run_phase4_full.sh 2>&1 | tee results/phase4_full.log

echo "===== DONE. Final decision table is at the end of results/phase4_full.log ====="
