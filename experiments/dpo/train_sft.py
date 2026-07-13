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
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "gpt2"
SFT_TEXTS = "results/phase4/sft_texts.json"
OUTPUT_DIR = "results/phase4/sft_base"
MAX_LENGTH = 512
SEED = 42


class KeepFirstEOSCollator:
    """Causal-LM collator for a tokenizer where pad_token_id == eos_token_id
    (GPT-2's standard workaround, since it has no dedicated pad token).

    Stock DataCollatorForLanguageModeling(mlm=False) masks every occurrence of
    pad_token_id out of the loss (labels[labels == pad_token_id] = -100). Since
    pad and eos share an ID here, that silently erases the loss signal for the
    ONE real end-of-response EOS token in every example too -- confirmed
    empirically: even 30 epochs of overfitting on tiny data never taught the
    model to predict EOS, because it never received gradient signal to do so.

    Fix: keep the FIRST eos_token_id per row as a live target (that's the real
    end-of-response marker); mask only positions AFTER it (genuine padding).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()
        eos_id = self.tokenizer.eos_token_id
        for i in range(labels.size(0)):
            row = labels[i]
            eos_positions = (row == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                if first_eos + 1 < len(row):
                    labels[i, first_eos + 1:] = -100
        batch["labels"] = labels
        return batch


def main():
    with open(SFT_TEXTS) as f:
        texts = json.load(f)
    print(f"SFT on {len(texts)} chosen transcripts")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tok.eos_token_id

    # Append EOS so the model has a training signal for "response is complete".
    # Without this, causal-LM SFT never sees an end-of-sequence token and the
    # model never learns to stop generating (confirmed: TRL's DPOTrainer does
    # NOT auto-append EOS either, so this must happen here, upstream of both
    # SFT and DPO, since DPO starts from this checkpoint).
    eos_texts = [t if t.endswith(tok.eos_token) else t + tok.eos_token for t in texts]

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = Dataset.from_dict({"text": eos_texts}).map(tok_fn, batched=True, remove_columns=["text"])
    collator = KeepFirstEOSCollator(tok)

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
