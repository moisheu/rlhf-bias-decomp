"""
Phase 4 shared SFT starting point: one brief causal-LM pass of GPT-2 over the
chosen transcripts of the SFT 5k (results/phase4/sft_texts.json). ALL DPO arms
start from this single checkpoint so arm differences aren't confounded by init.

Plain transformers Trainer + causal LM (mlm=False) — version-stable, no TRL SFT
API surface to track.

Run:  python -m experiments.dpo.train_sft
Env:  FORCE_CPU=1 to force CPU; SFT_EPOCHS (default 1).
"""
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "gpt2"
SFT_TEXTS = "results/phase4/sft_texts.json"
OUTPUT_DIR = "results/phase4/sft_base"
MAX_LENGTH = 512
SEED = 42


def main():
    with open(SFT_TEXTS) as f:
        texts = json.load(f)
    print(f"SFT on {len(texts)} chosen transcripts")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tok.eos_token_id

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = Dataset.from_dict({"text": texts}).map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=int(os.environ.get("SFT_EPOCHS", "1")),
        max_steps=int(os.environ.get("SFT_MAX_STEPS", "-1")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="no",
        bf16=False, fp16=False,
        use_cpu=os.environ.get("FORCE_CPU") == "1",
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    print("Training SFT...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved SFT base to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
