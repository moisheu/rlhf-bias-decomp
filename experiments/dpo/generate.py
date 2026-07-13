"""
Phase 4 generation + intrinsic metrics for one policy.

Eval prompts: 200 sampled (seed=42) from the 937-filtered eval set's Human-turn
side (disjoint from all training data by the upstream train/test split).

Generation is identical across policies: temperature 0.7, top-p 0.95, fixed
generation seed, max_new_tokens=256, one generation per prompt.

Per generation: completion token length (policy tokenizer), distinct-4-gram rate
(repetition guard), truncation flag (hit the cap). If RM dirs are given, also
cross-scores each generation (prompt+completion) under each RM (quality proxy).

Writes results/phase4/gen_{label}.json.

Run:
  python -m experiments.dpo.generate --policy-dir results/dpo_raw_seed42 --label raw_seed42 \
      --raw-rm results/reward_model_mixed_seed42 --reweight-rm results/reward_model_reweight_seed0
"""
import argparse
import json
import os
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data_utils import load_hh_rlhf_eval_subset

EVAL_N = 1000
EVAL_SEED = 42
RM_MAX_LENGTH = 512
N_PROMPTS = 200
GEN_SEED = 1234
SEP = "\n\nAssistant:"
WORD4 = 4


def get_device():
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_prompts(rm_tok):
    """937-filter the eval set (same as the RM eval), take Human-side prompts,
    sample 200 with seed=42."""
    ds = load_hh_rlhf_eval_subset(n=EVAL_N, split="test", seed=EVAL_SEED)
    eos = rm_tok.eos_token
    kept = []
    for row in ds:
        c, r = row["chosen"], row["rejected"]
        cids = rm_tok(c + eos)["input_ids"]
        rids = rm_tok(r + eos)["input_ids"]
        if len(cids) > RM_MAX_LENGTH or len(rids) > RM_MAX_LENGTH:
            continue
        prompt = c.rpartition(SEP)[0] + SEP
        kept.append(prompt)
    import random
    rng = random.Random(EVAL_SEED)
    rng.shuffle(kept)
    return kept[:N_PROMPTS]


def distinct_4gram_rate(token_ids):
    if len(token_ids) < WORD4:
        return 1.0
    grams = [tuple(token_ids[i:i + WORD4]) for i in range(len(token_ids) - WORD4 + 1)]
    return len(set(grams)) / len(grams)


@torch.no_grad()
def rm_score(texts, rm_tok, rm, device):
    eos = rm_tok.eos_token
    out = []
    for i in range(0, len(texts), 16):
        batch = [t + eos for t in texts[i:i + 16]]
        enc = rm_tok(batch, padding=True, truncation=True, max_length=RM_MAX_LENGTH,
                     return_tensors="pt").to(device)
        out.extend(rm(**enc).logits.squeeze(-1).cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--raw-rm", default=None)
    ap.add_argument("--reweight-rm", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    args = ap.parse_args()

    # Convert to absolute paths to avoid transformers' Hub ID validation
    args.policy_dir = os.path.abspath(args.policy_dir)
    if args.raw_rm:
        args.raw_rm = os.path.abspath(args.raw_rm)
    if args.reweight_rm:
        args.reweight_rm = os.path.abspath(args.reweight_rm)

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.policy_dir, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(args.policy_dir, local_files_only=True).eval().to(device)

    # RM tokenizer just for the 937-filter + prompt build (use raw-rm's, else policy's is wrong;
    # fall back to a distilbert tokenizer via the raw-rm dir if provided).
    rm_tok_for_filter = AutoTokenizer.from_pretrained(args.raw_rm) if args.raw_rm else tok
    if rm_tok_for_filter.eos_token is None:
        rm_tok_for_filter.eos_token = rm_tok_for_filter.sep_token
    prompts = build_prompts(rm_tok_for_filter)[: args.n_prompts]
    print(f"[{args.label}] generating for {len(prompts)} prompts on {device}")

    torch.manual_seed(GEN_SEED)
    records = []
    for idx, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
        gen = model.generate(
            **enc,
            do_sample=True, temperature=0.7, top_p=0.95,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
        # strip trailing pad/eos for length
        while new_ids and new_ids[-1] == tok.eos_token_id:
            new_ids.pop()
        completion = tok.decode(new_ids, skip_special_tokens=True)
        truncated = len(new_ids) >= args.max_new_tokens
        records.append({
            "prompt": prompt,
            "completion": completion,
            "gen_len": len(new_ids),
            "distinct4": distinct_4gram_rate(new_ids),
            "truncated": truncated,
        })
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(prompts)}", flush=True)

    # optional RM cross-scoring
    for name, rmdir in [("raw", args.raw_rm), ("reweight", args.reweight_rm)]:
        if not rmdir:
            continue
        rtok = AutoTokenizer.from_pretrained(rmdir, local_files_only=True)
        if rtok.eos_token is None:
            rtok.eos_token = rtok.sep_token
        rm = AutoModelForSequenceClassification.from_pretrained(rmdir, local_files_only=True).eval().to(device)
        fulltexts = [r["prompt"] + r["completion"] for r in records]
        scores = rm_score(fulltexts, rtok, rm, device)
        for r, s in zip(records, scores):
            r[f"rm_{name}_score"] = s
        del rm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    trunc_rate = sum(r["truncated"] for r in records) / len(records)
    os.makedirs("results/phase4", exist_ok=True)
    out = f"results/phase4/gen_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "policy_dir": args.policy_dir,
                   "n": len(records), "truncation_rate": trunc_rate,
                   "records": records}, f)
    import statistics
    print(f"[{args.label}] median gen_len={statistics.median(r['gen_len'] for r in records):.0f}  "
          f"mean distinct4={statistics.mean(r['distinct4'] for r in records):.3f}  "
          f"truncation={trunc_rate*100:.1f}%  -> {out}")
    if trunc_rate > 0.20:
        print(f"  WARNING: truncation {trunc_rate*100:.1f}% > 20% — raise --max-new-tokens and regenerate.")


if __name__ == "__main__":
    main()
