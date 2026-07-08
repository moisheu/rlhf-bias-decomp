"""
Length-bias evaluation harness for Phase 3 decomposition.

Scores a reward-model checkpoint on the FIXED benchmark eval set
(n=1000, seed=42, HH-RLHF test split) and reports the Chunk 9 table:
eval accuracy, eval margin, and length-score Pearson r (pooled / chosen-only
/ rejected-only).

Scoring replicates the TRL RewardTrainer pipeline exactly so the accuracy /
margin here match `trainer.evaluate()`:
  - append the tokenizer EOS to each response string (TRL `add_eos`),
  - tokenize with default special tokens,
  - drop any pair whose chosen OR rejected sequence exceeds max_length=512
    (TRL's `_prepare_dataset` filter) -> 937 of 1000 pairs survive,
  - score = model logit on the full Human/Assistant transcript.

Length definition (spec S2): tokenizer token count of the RESPONSE text (the
part after the final "\\n\\nAssistant:"), the principled "what the model sees"
choice. Full-sequence token length is reported alongside as a robustness line.

Run from the repo root:
    python -m experiments.decomposition.eval_length_correlation \
        --model-dir results/reward_model_mixed_seed42 --label mixed_seed42
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_utils import load_hh_rlhf_eval_subset

EVAL_N = 1000
EVAL_SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16


def get_device() -> torch.device:
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_response(text: str) -> str:
    return text.rsplit("\n\nAssistant:", 1)[-1].strip()


@torch.no_grad()
def score_texts(texts, tokenizer, model, device):
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        logits = model(**enc).logits.squeeze(-1)
        scores.extend(logits.detach().cpu().tolist())
    return scores


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_eval_records(tokenizer):
    """Reproduce TRL's eval pipeline: EOS-append, tokenize, filter >512 tokens."""
    eval_ds = load_hh_rlhf_eval_subset(n=EVAL_N, split="test", seed=EVAL_SEED)
    eos = tokenizer.eos_token

    records = []
    n_dropped = 0
    for row in eval_ds:
        chosen = row["chosen"]
        rejected = row["rejected"]
        chosen_eos = chosen if chosen.endswith(eos) else chosen + eos
        rejected_eos = rejected if rejected.endswith(eos) else rejected + eos

        # Full-sequence token ids WITH default special tokens (matches TRL
        # tokenize_fn: processing_class(text=...)['input_ids']). No truncation
        # here so the >512 filter sees true lengths.
        chosen_ids = tokenizer(text=chosen_eos)["input_ids"]
        rejected_ids = tokenizer(text=rejected_eos)["input_ids"]
        if len(chosen_ids) > MAX_LENGTH or len(rejected_ids) > MAX_LENGTH:
            n_dropped += 1
            continue

        # Response-only token length (spec S2): tokens of the text the model
        # sees for the response, excluding special tokens.
        resp_chosen = extract_response(chosen)
        resp_rejected = extract_response(rejected)
        rlen_chosen = len(tokenizer(resp_chosen, add_special_tokens=False)["input_ids"])
        rlen_rejected = len(tokenizer(resp_rejected, add_special_tokens=False)["input_ids"])

        records.append(
            {
                "chosen_text": chosen_eos,
                "rejected_text": rejected_eos,
                "seqlen_chosen": len(chosen_ids),
                "seqlen_rejected": len(rejected_ids),
                "rlen_chosen": rlen_chosen,
                "rlen_rejected": rlen_rejected,
            }
        )
    return records, n_dropped


def evaluate_checkpoint(model_dir, label):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    records, n_dropped = build_eval_records(tokenizer)
    n = len(records)

    chosen_scores = score_texts([r["chosen_text"] for r in records], tokenizer, model, device)
    rejected_scores = score_texts([r["rejected_text"] for r in records], tokenizer, model, device)

    diffs = np.array(chosen_scores) - np.array(rejected_scores)
    accuracy = float((diffs > 0).mean())
    margin = float(diffs.mean())

    rlen_chosen = [r["rlen_chosen"] for r in records]
    rlen_rejected = [r["rlen_rejected"] for r in records]
    seqlen_chosen = [r["seqlen_chosen"] for r in records]
    seqlen_rejected = [r["seqlen_rejected"] for r in records]

    # Response-length correlations (primary, spec S2)
    r_pooled = pearson(rlen_chosen + rlen_rejected, chosen_scores + rejected_scores)
    r_chosen = pearson(rlen_chosen, chosen_scores)
    r_rejected = pearson(rlen_rejected, rejected_scores)

    # Full-sequence-length correlations (robustness line)
    r_pooled_seq = pearson(seqlen_chosen + seqlen_rejected, chosen_scores + rejected_scores)

    result = {
        "label": label,
        "model_dir": model_dir,
        "n_eval": n,
        "n_dropped_over_maxlen": n_dropped,
        "accuracy": accuracy,
        "margin": margin,
        "r_pooled": r_pooled,
        "r_chosen": r_chosen,
        "r_rejected": r_rejected,
        "r_pooled_seqlen": r_pooled_seq,
        "mean_reward": float((np.array(chosen_scores + rejected_scores)).mean()),
        "std_reward": float((np.array(chosen_scores + rejected_scores)).std()),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON path to append the result to.")
    args = parser.parse_args()

    result = evaluate_checkpoint(args.model_dir, args.label)

    print(json.dumps(result, indent=2))
    print(
        f"\n[{args.label}] n={result['n_eval']} (dropped {result['n_dropped_over_maxlen']})  "
        f"acc={result['accuracy']:.4f}  margin={result['margin']:.4f}  "
        f"r_pooled={result['r_pooled']:.4f}  r_chosen={result['r_chosen']:.4f}  "
        f"r_rejected={result['r_rejected']:.4f}  (r_pooled_seqlen={result['r_pooled_seqlen']:.4f})"
    )

    if args.out:
        existing = []
        if os.path.exists(args.out):
            with open(args.out) as f:
                existing = json.load(f)
        existing = [r for r in existing if r.get("label") != args.label]
        existing.append(result)
        with open(args.out, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Appended to {args.out}")


if __name__ == "__main__":
    main()
