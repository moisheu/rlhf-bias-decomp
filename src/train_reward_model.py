"""
Train a baseline reward model on a subset of HH-RLHF, with early stopping
on held-out eval accuracy (not train loss).

Run from the repo root:
    FORCE_CPU=1 python -m src.train_reward_model
    FORCE_CPU=1 DATA_MODE=mixed python -m src.train_reward_model
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from trl import RewardConfig, RewardTrainer

from src.data_utils import load_hh_rlhf_subset, load_hh_rlhf_mixed, load_hh_rlhf_eval_subset

MODEL_NAME = "distilbert-base-uncased"
SEED = int(os.environ.get("TRAIN_SEED", "42"))
# "harmless" = load_hh_rlhf_subset's unshuffled harmless-base-only prefix
# (original Phase 1 pool); "mixed" = load_hh_rlhf_mixed, shuffled across all
# four subsets before truncating, seeded on TRAIN_SEED so each seed draws its
# own (slightly different) mixed pool.
DATA_MODE = os.environ.get("DATA_MODE", "harmless")
if DATA_MODE == "mixed":
    OUTPUT_DIR = f"results/reward_model_mixed_seed{SEED}"
else:
    OUTPUT_DIR = "results/reward_model_baseline" if SEED == 42 else f"results/reward_model_baseline_seed{SEED}"
N_TRAIN = 30000

# Fixed benchmark eval set: n=1000, seed=42, HH-RLHF test split. Do not change
# these two values in future runs (including bias-corrected variants) —
# keeping them fixed is what makes eval numbers comparable across runs.
EVAL_N = 1000
EVAL_SEED = 42


def main():
    print(f"Loading dataset (DATA_MODE={DATA_MODE})...")
    if DATA_MODE == "mixed":
        dataset = load_hh_rlhf_mixed(n=N_TRAIN, seed=SEED)
    else:
        dataset = load_hh_rlhf_subset(n=N_TRAIN)
    print(f"  {len(dataset)} train examples, columns: {dataset.column_names}")

    eval_dataset = load_hh_rlhf_eval_subset(n=EVAL_N, split="test", seed=EVAL_SEED)
    print(f"  {len(eval_dataset)} held-out eval examples (test split, seed={EVAL_SEED})")

    print(f"Loading model: {MODEL_NAME} (seed={SEED})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # DistilBERT has no EOS token; TRL requires one to append to sequences
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    # Seed before model creation so the randomly initialized classification
    # head (not just downstream training stochasticity) is seed-controlled.
    torch.manual_seed(SEED)
    # num_labels=1 → scalar reward head (single logit per sequence)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        use_cpu=os.environ.get("FORCE_CPU") == "1",
        # RewardConfig defaults bf16=True, which silently falls back to an
        # unoptimized scalar GEMM kernel on Apple Accelerate BLAS (CPU) —
        # confirmed via profiling to be the cause of a >40x slowdown. Keep
        # both off on this hardware.
        bf16=False,
        fp16=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
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
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # TRL >= 0.8 API
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Training...")
    trainer.train()

    print("Final held-out eval (best checkpoint, restored via load_best_model_at_end)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
