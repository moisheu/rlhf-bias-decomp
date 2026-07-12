#!/usr/bin/env bash
# One-shot setup for a fresh RunPod/SSH GPU box. Pulls the repo onto the
# correct branch, installs pinned deps, and kicks off the full Phase 4
# pipeline. Meant to be curl'd directly onto the box (see instructions below)
# so nothing has to be copy-pasted through a terminal that may mangle quotes.
#
# Usage on the pod:
#   curl -H "Authorization: token <YOUR_GITHUB_TOKEN>" \
#     -o pod_setup.sh \
#     https://raw.githubusercontent.com/moisheu/rlhf-bias-decomp/phase3-decomposition/experiments/dpo/pod_setup.sh
#   GITHUB_TOKEN=<YOUR_GITHUB_TOKEN> bash pod_setup.sh
set -eu

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN=<your token> before running this script}"
REPO_DIR="${REPO_DIR:-/rlhf-bias-decomp}"
BRANCH=phase3-decomposition
REPO_URL="https://${GITHUB_TOKEN}@github.com/moisheu/rlhf-bias-decomp.git"

echo "===== Clone / checkout ====="
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

echo "===== Cleanup any stray files from earlier paste mishaps ====="
rm -f '=0.27.0' 2>/dev/null || true

echo "===== Install pinned dependencies ====="
pip install -q transformers==5.9.0 trl==1.5.0 datasets accelerate scipy numpy

echo "===== Sanity check ====="
ls experiments/dpo/run_phase4_full.sh && echo "run_phase4_full.sh found, good to go."

echo ""
echo "===== Setup complete. Now run the pipeline (inside tmux!): ====="
echo "  tmux new -s phase4"
echo "  cd $REPO_DIR && bash experiments/dpo/run_phase4_full.sh 2>&1 | tee results/phase4_full.log"
