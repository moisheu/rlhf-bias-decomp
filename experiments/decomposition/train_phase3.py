"""
Phase 3 decomposition training driver.

One controlled change at a time on top of the mixed baseline. Every arm trains
on the IDENTICAL 30k rows: the seed=42 mixed pool (load_hh_rlhf_mixed(n=30000,
seed=42)) for every arm and every training seed (spec S3). Only the model /
training seed varies across seeds; the pool never does. This is what makes the
method effect un-confounded by pool resampling.

Methods (METHOD env var):
  symloss   : symmetric sigmoid loss              (Method 2)
  reweight  : per-pair importance weights         (Method 1)
  combined  : symmetric loss + weights            (combined arm, promoted core)

Env vars:
  METHOD      one of {symloss, reweight, combined}   (required)
  TRAIN_SEED  model/training seed, default 42
  FORCE_CPU   "1" to force CPU (else MPS/CUDA auto)

Run from the repo root:
  METHOD=symloss  TRAIN_SEED=42 python -m experiments.decomposition.train_phase3
  METHOD=reweight TRAIN_SEED=42 python -m experiments.decomposition.train_phase3
  METHOD=combined TRAIN_SEED=42 python -m experiments.decomposition.train_phase3
"""
import json
import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import RewardConfig

from src.data_utils import load_hh_rlhf_mixed, load_hh_rlhf_eval_subset
from experiments.decomposition.weighted_trainer import (
    Phase3RewardTrainer,
    DataCollatorForWeightedPreference,
)

MODEL_NAME = "distilbert-base-uncased"
METHOD = os.environ["METHOD"]
SEED = int(os.environ.get("TRAIN_SEED", "42"))
assert METHOD in {"symloss", "reweight", "combined"}, METHOD

# Fixed training pool: the seed=42 mixed sample, for EVERY arm/seed (spec S3).
POOL_SEED = 42
N_TRAIN = 30000
WEIGHTS_PATH = "results/reweight_weights_seed42pool.json"

OUTPUT_DIR = f"results/reward_model_{METHOD}_seed{SEED}"

# Fixed benchmark eval set (unchanged from baseline; unweighted).
EVAL_N = 1000
EVAL_SEED = 42

SYMMETRIC = METHOD in {"symloss", "combined"}
USE_WEIGHTS = METHOD in {"reweight", "combined"}

# Effective train batch is always 16 (matches baseline); TRAIN_BATCH sets the
# micro-batch and gradient accumulation makes up the rest. Lower micro-batch =
# lower MPS peak memory (this machine's GPU is shared with other apps). Dropout
# is disabled, so any micro-batch size is forward-equivalent to batch 16.
EFF_BATCH = 16
MICRO_BATCH = int(os.environ.get("TRAIN_BATCH", "4"))
GRAD_ACCUM = max(1, EFF_BATCH // MICRO_BATCH)
# Symmetric loss can converge more slowly -> allow up to 2x epochs with the
# same early-stopping rule (spec Method 2 training notes). One change at a time:
# do not touch LR.
NUM_EPOCHS = 6 if SYMMETRIC else 3


def main():
    print(f"=== Phase 3 arm: METHOD={METHOD} seed={SEED} (symmetric={SYMMETRIC} weights={USE_WEIGHTS}) ===")
    print(f"Loading FIXED seed={POOL_SEED} mixed pool (n={N_TRAIN})...")
    dataset = load_hh_rlhf_mixed(n=N_TRAIN, seed=POOL_SEED)
    print(f"  {len(dataset)} train examples, columns: {dataset.column_names}")

    if USE_WEIGHTS:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"{WEIGHTS_PATH} not found. Run compute_weights.py first."
            )
        with open(WEIGHTS_PATH) as f:
            wdata = json.load(f)
        weights = wdata["weights"]  # list indexed by pool row index
        if len(weights) != len(dataset):
            raise ValueError(
                f"weights length {len(weights)} != pool length {len(dataset)}; "
                "pool/weights mismatch (wrong seed or n)."
            )
        # Attach BEFORE tokenization so it survives .map/.filter (spec plumbing).
        dataset = dataset.add_column("weight", [float(w) for w in weights])
        print(f"  attached weight column (mean={sum(weights)/len(weights):.4f})")

    eval_dataset = load_hh_rlhf_eval_subset(n=EVAL_N, split="test", seed=EVAL_SEED)
    print(f"  {len(eval_dataset)} held-out eval examples (unweighted)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    # Seed before model creation so the random classification head is controlled.
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        use_cpu=os.environ.get("FORCE_CPU") == "1",
        bf16=False,
        fp16=False,
        # Memory-driven, dynamics-preserving (see EFF_BATCH/MICRO_BATCH above).
        per_device_train_batch_size=MICRO_BATCH,
        per_device_eval_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        max_length=512,
        logging_steps=250,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
        # CRITICAL: keep the `weight` column alive through to the collator
        # (spec failure-mode guard #1). Harmless for the no-weight arms.
        remove_unused_columns=False,
    )

    collator = DataCollatorForWeightedPreference(pad_token_id=tokenizer.pad_token_id or tokenizer.sep_token_id)

    trainer = Phase3RewardTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        symmetric=SYMMETRIC,
    )

    print("Training...")
    trainer.train()

    print("Final held-out eval (best checkpoint restored)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
