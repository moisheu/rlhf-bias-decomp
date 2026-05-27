"""
Train a baseline reward model on a 5k subset of HH-RLHF.

Run from the repo root:
    python -m src.train_reward_model
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from src.data_utils import load_hh_rlhf_subset

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "results/reward_model_baseline"


def main():
    print("Loading dataset...")
    dataset = load_hh_rlhf_subset(n=5000)
    print(f"  {len(dataset)} examples, columns: {dataset.column_names}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # DistilBERT has no EOS token; TRL requires one to append to sequences
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    # num_labels=1 → scalar reward head (single logit per sequence)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        max_length=512,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,  # TRL >= 0.8 API
    )

    print("Training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
