#!/usr/bin/env bash
# Verify enough MPS headroom exists, then launch the queue exactly once.
# A Metal driver-level leak from earlier OOM crashes holds ~13 GiB with no
# owning process (only a reboot frees it), leaving ~7 GiB usable. batch-2 peaks
# ~4 GiB, so we require a 5 GiB probe to pass before launching. TRAIN_BATCH can
# be raised (and PROBE_GB with it) after a reboot reclaims the full GPU.
set -u
cd "$(dirname "$0")/../.."

: "${TRAIN_BATCH:=2}"
: "${PROBE_GB:=5}"
export TRAIN_BATCH

echo "[wait_and_launch] $(date) TRAIN_BATCH=$TRAIN_BATCH probing ${PROBE_GB}GB headroom..."
for i in $(seq 1 15); do
  free=$(PROBE_GB=$PROBE_GB python - <<'PY'
import os, torch
gb=float(os.environ["PROBE_GB"])
try:
    x=torch.empty(int(gb*1e9//4), dtype=torch.float32, device="mps"); torch.mps.synchronize()
    del x; torch.mps.empty_cache(); print("FREE")
except RuntimeError:
    print("BUSY")
PY
)
  echo "[wait_and_launch] attempt $i: ${PROBE_GB}GB headroom $free"
  if [ "$free" = "FREE" ]; then
    echo "[wait_and_launch] launching queue (TRAIN_BATCH=$TRAIN_BATCH)"
    exec bash experiments/decomposition/run_queue.sh
  fi
  sleep 20
done
echo "[wait_and_launch] insufficient headroom after 5 min — not launching."
exit 1
