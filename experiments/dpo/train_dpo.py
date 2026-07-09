"""
Phase 4 DPO: fine-tune the shared SFT base on RM-relabeled (or human) preference
pairs. The only thing that varies across arms is which labeled file is used, so
downstream behavioral differences are attributable to the labeler's bias.

TRL 1.5.0 DPOTrainer/DPOConfig. GPT-2 policy + a frozen GPT-2 reference, both
from results/phase4/sft_base. beta=0.1 fixed (scope limit). Explicit-prompt
dataset (prompt/chosen/rejected).

Kill-switch: a callback flags NaN/inf training loss. On instability the script
exits 2 so the orchestrator can retry ONCE at half LR; a second instability
means stop (do not tune) per the spec.

Run:
  LABELER=raw TRAIN_SEED=42 python -m experiments.dpo.train_dpo
Env: LABELER in {raw,human,reweight,corr}; TRAIN_SEED; DPO_LR (default 5e-6);
     FORCE_CPU=1; DPO_MAX_STEPS (smoke tests).
"""
import json
import math
import os
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

SFT_BASE = "results/phase4/sft_base"
LABELER = os.environ["LABELER"]
SEED = int(os.environ.get("TRAIN_SEED", "42"))
LR = float(os.environ.get("DPO_LR", "5e-6"))
BETA = 0.1
MAX_LENGTH = 512
OUTPUT_DIR = f"results/dpo_{LABELER}_seed{SEED}"
LABELED = f"results/phase4/dpo_labeled_{LABELER}.json"


class InstabilityCallback(TrainerCallback):
    """Flag NaN/inf loss (the kill-switch signal)."""
    def __init__(self):
        self.unstable = False
        self.last_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            v = logs["loss"]
            self.last_loss = v
            if v is None or math.isnan(v) or math.isinf(v):
                self.unstable = True


def main():
    with open(LABELED) as f:
        rows = json.load(f)
    ds = Dataset.from_list(rows)  # columns: prompt, chosen, rejected
    print(f"[DPO] labeler={LABELER} seed={SEED} lr={LR} on {len(ds)} pairs")

    tok = AutoTokenizer.from_pretrained(SFT_BASE)
    tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(SFT_BASE)
    ref = AutoModelForCausalLM.from_pretrained(SFT_BASE)
    model.config.pad_token_id = tok.eos_token_id
    ref.config.pad_token_id = tok.eos_token_id

    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        num_train_epochs=1,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,   # effective 8
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        bf16=False, fp16=False,
        use_cpu=os.environ.get("FORCE_CPU") == "1",
        seed=SEED,
        report_to="none",
        max_grad_norm=1.0,
        pad_token=tok.eos_token,
        max_steps=int(os.environ.get("DPO_MAX_STEPS", "-1")),
    )

    cb = InstabilityCallback()
    trainer = DPOTrainer(
        model=model,
        ref_model=ref,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[cb],
    )
    print("Training DPO...")
    trainer.train()

    if cb.unstable:
        print(f"[DPO] UNSTABLE (NaN/inf loss) at lr={LR} — not saving.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"[DPO] stable (last loss {cb.last_loss}). Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
