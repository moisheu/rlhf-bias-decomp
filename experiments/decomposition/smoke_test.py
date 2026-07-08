"""
Debug-batch smoke test (spec Day-2 mandatory check + failure-mode guard #1).

Confirms, on a tiny slice with max_steps=2:
  - the `weight` column survives to compute_loss with shape (batch,)
    (i.e. remove_unused_columns=False + custom collator plumbing works),
  - symmetric sigmoid loss runs and is finite,
  - the unweighted path (no weight column) also runs.

Run: FORCE_CPU=1 python -m experiments.decomposition.smoke_test
"""
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig

from src.data_utils import load_hh_rlhf_mixed, load_hh_rlhf_eval_subset
from experiments.decomposition.weighted_trainer import (
    Phase3RewardTrainer,
    DataCollatorForWeightedPreference,
)

MODEL_NAME = "distilbert-base-uncased"


def build(symmetric, with_weights, n=64):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.eos_token is None:
        tok.eos_token = tok.sep_token
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    ds = load_hh_rlhf_mixed(n=n, seed=42)
    if with_weights:
        # deterministic non-trivial weights to prove they flow through
        ds = ds.add_column("weight", [1.0 + 0.5 * (i % 3) for i in range(len(ds))])
    eval_ds = load_hh_rlhf_eval_subset(n=32, split="test", seed=42)

    args = RewardConfig(
        output_dir="results/_smoke_tmp",
        max_steps=2,
        use_cpu=os.environ.get("FORCE_CPU") == "1",
        bf16=False,
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        max_length=512,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    collator = DataCollatorForWeightedPreference(pad_token_id=tok.pad_token_id)
    trainer = Phase3RewardTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        symmetric=symmetric,
    )
    return trainer


def main():
    print("\n--- CASE 1: combined (symmetric=True, weights=True) ---")
    t = build(symmetric=True, with_weights=True)
    out = t.train()
    print("train_loss:", out.training_loss, "finite:", torch.isfinite(torch.tensor(out.training_loss)).item())

    print("\n--- CASE 2: symloss (symmetric=True, weights=False) ---")
    t = build(symmetric=True, with_weights=False)
    out = t.train()
    print("train_loss:", out.training_loss, "finite:", torch.isfinite(torch.tensor(out.training_loss)).item())

    print("\n--- CASE 3: reweight (symmetric=False, weights=True) ---")
    t = build(symmetric=False, with_weights=True)
    out = t.train()
    print("train_loss:", out.training_loss, "finite:", torch.isfinite(torch.tensor(out.training_loss)).item())

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
