"""
Phase 4 CPU smoke test: exercises SFT (causal LM), DPO (TRL DPOTrainer/DPOConfig
1.5.0), and generation on tiny data in a tmp dir, to catch API/version bugs
before spending Colab GPU time. Not a real run.

Run: FORCE_CPU=1 python -m experiments.dpo.smoke_test
"""
import os
import shutil

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

TMP = "results/phase4/_smoke"


def tiny_sft():
    print("--- SFT (2 steps) ---")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    texts = ["\n\nHuman: hi\n\nAssistant: hello there, how can I help?"] * 16
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=128), batched=True,
        remove_columns=["text"])
    args = TrainingArguments(output_dir=f"{TMP}/sft", max_steps=2,
                             per_device_train_batch_size=4, use_cpu=True,
                             bf16=False, fp16=False, report_to="none", logging_steps=1)
    tr = Trainer(model=model, args=args, train_dataset=ds,
                 data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))
    tr.train()
    tr.save_model(f"{TMP}/sft"); tok.save_pretrained(f"{TMP}/sft")
    print("  SFT ok")


def tiny_dpo():
    print("--- DPO (2 steps) ---")
    tok = AutoTokenizer.from_pretrained(f"{TMP}/sft")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"{TMP}/sft")
    ref = AutoModelForCausalLM.from_pretrained(f"{TMP}/sft")
    rows = [{"prompt": "\n\nHuman: hi\n\nAssistant:",
             "chosen": " hello, happy to help with anything you need today.",
             "rejected": " no."}] * 16
    ds = Dataset.from_list(rows)
    args = DPOConfig(output_dir=f"{TMP}/dpo", beta=0.1, max_steps=2, max_length=256,
                     per_device_train_batch_size=2, gradient_accumulation_steps=1,
                     learning_rate=5e-6, use_cpu=True, bf16=False, fp16=False,
                     report_to="none", logging_steps=1, pad_token=tok.eos_token)
    tr = DPOTrainer(model=model, ref_model=ref, args=args, train_dataset=ds,
                    processing_class=tok)
    tr.train()
    tr.save_model(f"{TMP}/dpo"); tok.save_pretrained(f"{TMP}/dpo")
    print("  DPO ok")


def tiny_generate():
    print("--- generate ---")
    tok = AutoTokenizer.from_pretrained(f"{TMP}/dpo")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(f"{TMP}/dpo").eval()
    enc = tok("\n\nHuman: what's the weather?\n\nAssistant:", return_tensors="pt")
    torch.manual_seed(1234)
    out = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95,
                         max_new_tokens=16, pad_token_id=tok.eos_token_id)
    new = out[0][enc["input_ids"].shape[1]:]
    print("  gen:", repr(tok.decode(new, skip_special_tokens=True))[:80])
    print("  generate ok")


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    tiny_sft()
    tiny_dpo()
    tiny_generate()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nPHASE 4 SMOKE TEST OK")


if __name__ == "__main__":
    main()
