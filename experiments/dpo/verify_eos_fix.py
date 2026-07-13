"""
Verify the EOS fix actually teaches GPT-2 to stop generating, before spending
GPU time on a full retrain. Two stages, mirroring the real pipeline:
  1. SFT (train_sft.py's fix: EOS-appended texts + KeepFirstEOSCollator) --
     checks generation stops naturally before hitting max_new_tokens.
  2. DPO on top of that checkpoint (relabel.py's fix: EOS-appended chosen/
     rejected) -- checks DPOTrainer doesn't ALSO neutralize the stopping
     behavior somehow (it uses a different loss mechanism than SFT's causal-LM
     labels, so this isn't guaranteed by the SFT fix alone).

Run: FORCE_CPU=1 python -m experiments.dpo.verify_eos_fix
"""
import shutil

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DPOConfig, DPOTrainer

from experiments.dpo.train_sft import KeepFirstEOSCollator

TMP = "results/phase4/_verify_eos_tmp"
MAX_NEW_TOKENS = 100

# Diverse, EOS-terminated examples -- a single repeated identical line at a
# high LR (an earlier version of this test) catastrophically overfits into a
# repetition loop, a different failure mode than "never learned EOS" and was
# misleading. Diversity + a real-ish LR is the fair test.
BASE_PAIRS = [
    ("hi", "hello! how can I help you today?"),
    ("what's 2+2?", "2+2 equals 4."),
    ("tell me a joke", "why did the chicken cross the road? to get to the other side!"),
    ("what's the weather?", "I don't have real-time weather data, sorry."),
    ("who are you?", "I'm an AI assistant here to help."),
    ("thanks", "you're welcome!"),
    ("bye", "goodbye, have a great day!"),
    ("help me code", "sure, what language are you working in?"),
]


def check_stopping(model, tok, label):
    model.eval()
    results = []
    for h, _ in BASE_PAIRS:
        enc = tok(f"\n\nHuman: {h}\n\nAssistant:", return_tensors="pt")
        torch.manual_seed(hash(h) % 10000)
        gen = model.generate(
            **enc, do_sample=True, temperature=0.7, top_p=0.95,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
        new_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
        hit_cap = len(new_ids) >= MAX_NEW_TOKENS
        ended_with_eos = tok.eos_token_id in new_ids
        results.append((h, len(new_ids), ended_with_eos, hit_cap))
        print(f"  [{h!r}] {len(new_ids)} tokens, eos={ended_with_eos}, hit_cap={hit_cap}: "
              f"{tok.decode(new_ids)[:60]!r}")
    n_stopped = sum(1 for _, _, eos, cap in results if eos and not cap)
    print(f"  -> {label}: {n_stopped}/{len(results)} stopped naturally with EOS")
    return n_stopped >= len(results) * 0.75


def run_sft():
    print("=" * 70)
    print("STAGE 1: SFT with the fix (EOS-appended texts + KeepFirstEOSCollator)")
    print("=" * 70)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = tok.eos_token_id

    texts = [
        f"\n\nHuman: {h}\n\nAssistant: {a}" + tok.eos_token
        for h, a in BASE_PAIRS
        for _ in range(8)
    ]

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=64)

    ds = Dataset.from_dict({"text": texts}).map(tok_fn, batched=True, remove_columns=["text"])
    collator = KeepFirstEOSCollator(tok)

    args = TrainingArguments(
        output_dir=f"{TMP}/sft", num_train_epochs=30, per_device_train_batch_size=8,
        learning_rate=5e-5, use_cpu=True, bf16=False, fp16=False,
        logging_steps=20, report_to="none", save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    sft_ok = check_stopping(model, tok, "SFT")
    trainer.save_model(f"{TMP}/sft")
    tok.save_pretrained(f"{TMP}/sft")
    return sft_ok


def run_dpo():
    print("\n" + "=" * 70)
    print("STAGE 2: DPO on top of the fixed SFT checkpoint (EOS-appended chosen/rejected)")
    print("=" * 70)
    tok = AutoTokenizer.from_pretrained(f"{TMP}/sft")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(f"{TMP}/sft")
    ref = AutoModelForCausalLM.from_pretrained(f"{TMP}/sft")
    model.config.pad_token_id = tok.eos_token_id
    ref.config.pad_token_id = tok.eos_token_id

    # DPO pairs: chosen is a couple tokens longer than rejected (mimics a mild
    # length preference), both EOS-terminated exactly like relabel.py's fix.
    rows = []
    for h, a in BASE_PAIRS:
        chosen = a + " let me know if there's anything else!" + tok.eos_token
        rejected = a.split(".")[0].split("!")[0] + "." + tok.eos_token
        for _ in range(4):
            rows.append({"prompt": f"\n\nHuman: {h}\n\nAssistant:", "chosen": chosen, "rejected": rejected})
    ds = Dataset.from_list(rows)

    args = DPOConfig(
        output_dir=f"{TMP}/dpo", beta=0.1, num_train_epochs=15, max_length=128,
        per_device_train_batch_size=8, learning_rate=5e-6, use_cpu=True,
        bf16=False, fp16=False, report_to="none", logging_steps=10,
        save_strategy="no", pad_token=tok.eos_token,
    )
    trainer = DPOTrainer(model=model, ref_model=ref, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()

    return check_stopping(model, tok, "DPO")


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    sft_ok = run_sft()
    dpo_ok = run_dpo()
    shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"SFT stopping behavior: {'PASS' if sft_ok else 'FAIL'}")
    print(f"DPO stopping behavior: {'PASS' if dpo_ok else 'FAIL'}")
    print("OVERALL:", "PASS -- safe to retrain the full pipeline" if (sft_ok and dpo_ok)
          else "FAIL -- do not spend GPU time yet, more investigation needed")


if __name__ == "__main__":
    main()
