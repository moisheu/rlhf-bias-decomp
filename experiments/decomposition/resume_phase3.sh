#!/usr/bin/env bash
# Resume the Phase 3 run queue after a reboot (which reclaims the leaked GPU
# memory). Safe to run repeatedly: arms with a .done marker are skipped, so it
# continues where it left off. With a freshly rebooted (clean) GPU, batch-4
# peaks ~7 GiB against ~18 GiB free -> comfortable margin, no OOM, no re-leak.
#
# Usage (from repo root):
#   bash experiments/decomposition/resume_phase3.sh
set -eu
cd "$(dirname "$0")/../.."

source ~/.venvs/rlhf-bias/bin/activate

# Confirm the GPU actually reclaimed (expect a large allocation to succeed).
python - <<'PY'
import torch
try:
    x=torch.empty(int(15e9//4), dtype=torch.float32, device="mps"); torch.mps.synchronize()
    del x; torch.mps.empty_cache(); print("[resume] GPU clean: 15GB allocation OK")
except RuntimeError:
    print("[resume] WARNING: GPU still constrained — reboot may not have completed. "
          "Proceeding at batch-2; check results/phase3_logs/queue_status.txt")
PY

TRAIN_BATCH=4 PROBE_GB=10 nohup bash experiments/decomposition/wait_and_launch.sh \
    > results/phase3_logs/wait_launch.log 2>&1 &
echo "[resume] queue launcher started (PID $!). Monitor: tail -f results/phase3_logs/queue_status.txt"
