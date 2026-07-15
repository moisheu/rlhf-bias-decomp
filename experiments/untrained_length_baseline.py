"""
Untrained-model length-correlation baseline: the pre-training control for the
"training induces length bias" claim.

Scores a FRESHLY-INITIALIZED distilbert-base-uncased scalar head (no training)
on the same fixed benchmark eval set as the trained runs, for each seed in
{42, 0, 1} -- the same seeds as the trained mixed-pool runs, so every seed can
be compared paired rather than only population-to-population.

The eval-record construction, scoring, and Pearson computation are IMPORTED from
experiments.decomposition.eval_length_correlation rather than reimplemented, so
these numbers are computed by exactly the same code path that produced
results/phase3_length_corr.json for the trained checkpoints (same EOS-append,
same >512-token pair filter -> 937 of 1000 pairs, same response-length
definition from spec S2).

Determinism: torch.manual_seed(seed) is set immediately before from_pretrained,
which is what makes the randomly-initialized classification head reproducible --
this mirrors src/train_reward_model.py, which seeds at the same point for the
same reason. Runs on CPU by default: the head init and the forward pass are then
bit-reproducible across machines, which matters for a number that goes in a
paper. --device mps/cuda is available but not guaranteed bit-identical.

Run from the repo root:
    python -m experiments.untrained_length_baseline
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.decomposition.eval_length_correlation import (
    build_eval_records,
    pearson,
    score_texts,
)

MODEL_NAME = "distilbert-base-uncased"
SEEDS = [42, 0, 1]
OUT_PATH = "results/untrained_length_correlation.json"


def untrained_correlations(seed, tokenizer, records, device):
    """Pooled / chosen-only / rejected-only length-score Pearson r for one
    freshly-initialized (untrained) scalar head."""
    # Seed BEFORE model creation so the random classification head is the thing
    # under seed control (matches src/train_reward_model.py's ordering).
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.eval()
    model.to(device)

    chosen_scores = score_texts([r["chosen_text"] for r in records], tokenizer, model, device)
    rejected_scores = score_texts([r["rejected_text"] for r in records], tokenizer, model, device)

    rlen_chosen = [r["rlen_chosen"] for r in records]
    rlen_rejected = [r["rlen_rejected"] for r in records]

    return {
        "pooled_r": pearson(rlen_chosen + rlen_rejected, chosen_scores + rejected_scores),
        "chosen_r": pearson(rlen_chosen, chosen_scores),
        "rejected_r": pearson(rlen_rejected, rejected_scores),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu (default, reproducible) / mps / cuda")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token

    # Built once: identical across seeds (depends only on the tokenizer), and
    # identical to the trained runs' eval set.
    records, n_dropped = build_eval_records(tokenizer)
    print(f"eval set: {len(records)} pairs kept, {n_dropped} dropped over max_length=512")

    results = {}
    for seed in SEEDS:
        results[str(seed)] = untrained_correlations(seed, tokenizer, records, device)
        r = results[str(seed)]
        print(
            f"[untrained seed={seed}] pooled_r={r['pooled_r']:+.4f}  "
            f"chosen_r={r['chosen_r']:+.4f}  rejected_r={r['rejected_r']:+.4f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
