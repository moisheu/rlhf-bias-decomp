"""
Phase 4 relabeling: score each DPO pair's two completions with a reward model
and keep the RM-preferred side as "chosen". The ONLY difference between DPO arms
is which RM (or human) produced the labels, so any downstream behavioral gap is
attributable to the RMs' differing biases.

For labeler == "human", the original human labels are kept unchanged (anchor).

Writes results/phase4/dpo_labeled_{labeler}.json: list[{prompt, chosen, rejected}]
(explicit-prompt DPO format) and prints agreement-with-human stats. When a second
labeled file exists, also reports pairwise RM disagreement and saves the mask.

Run:
  # human anchor (no RM):
  python -m experiments.dpo.relabel --labeler human
  # RM relabelers:
  python -m experiments.dpo.relabel --labeler raw      --model-dir results/reward_model_mixed_seed42
  python -m experiments.dpo.relabel --labeler reweight --model-dir results/reward_model_reweight_seed0
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

IN_PAIRS = "results/phase4/dpo_pairs.json"
OUT_DIR = "results/phase4"
MAX_LENGTH = 512
BATCH_SIZE = 16

# GPT-2's EOS string. Appended to every chosen/rejected completion written out
# here (independent of whatever RM tokenizer is used above for scoring) so the
# downstream DPO policy has a training signal for "response is complete" --
# TRL 1.5.0's DPOTrainer does NOT auto-append EOS, so without this the policy
# never learns to stop generating and just fills whatever max_new_tokens is set
# to, regardless of its value (confirmed empirically: doubling the cap 256->512
# didn't reduce truncation at all, since every generation just filled the cap).
POLICY_EOS = "<|endoftext|>"


def with_eos(text: str) -> str:
    return text if text.endswith(POLICY_EOS) else text + POLICY_EOS


def get_device():
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score(texts, tok, model, device):
    eos = tok.eos_token
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t + eos for t in texts[i:i + BATCH_SIZE]]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                  return_tensors="pt").to(device)
        out.extend(model(**enc).logits.squeeze(-1).cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", required=True, choices=["human", "raw", "reweight", "corr"])
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    with open(IN_PAIRS) as f:
        pairs = json.load(f)
    n = len(pairs)

    # human_chosen is, by construction, the human-preferred side.
    if args.labeler == "human":
        labeled = [{"prompt": p["prompt"], "chosen": with_eos(p["human_chosen"]),
                    "rejected": with_eos(p["human_rejected"])} for p in pairs]
        rm_prefers_human = [True] * n
        print(f"[human] kept original labels for {n} pairs.")
    else:
        assert args.model_dir, "RM labeler needs --model-dir"
        device = get_device()
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        if tok.eos_token is None:
            tok.eos_token = tok.sep_token
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval().to(device)

        # score full transcripts (prompt + completion), as the RM was trained
        chosen_full = [p["prompt"] + p["human_chosen"] for p in pairs]
        rejected_full = [p["prompt"] + p["human_rejected"] for p in pairs]
        s_hc = score(chosen_full, tok, model, device)
        s_hr = score(rejected_full, tok, model, device)

        labeled, rm_prefers_human = [], []
        for p, a, b in zip(pairs, s_hc, s_hr):
            prefers_human = a >= b  # RM prefers the human-chosen side
            rm_prefers_human.append(prefers_human)
            if prefers_human:
                labeled.append({"prompt": p["prompt"], "chosen": with_eos(p["human_chosen"]),
                                "rejected": with_eos(p["human_rejected"])})
            else:
                labeled.append({"prompt": p["prompt"], "chosen": with_eos(p["human_rejected"]),
                                "rejected": with_eos(p["human_chosen"])})
        agree = sum(rm_prefers_human) / n
        print(f"[{args.labeler}] RM={args.model_dir}")
        print(f"  agreement with human labels: {agree*100:.2f}%  "
              f"(flipped {n - sum(rm_prefers_human)} of {n} pairs)")

    out_path = f"{OUT_DIR}/dpo_labeled_{args.labeler}.json"
    with open(out_path, "w") as f:
        json.dump(labeled, f)
    # save the RM's per-pair preference mask for cross-arm disagreement analysis
    with open(f"{OUT_DIR}/prefmask_{args.labeler}.json", "w") as f:
        json.dump(rm_prefers_human, f)
    print(f"  wrote {out_path}")

    # pairwise RM disagreement vs any other saved RM mask
    for other in ["raw", "reweight", "corr"]:
        if other == args.labeler:
            continue
        mpath = f"{OUT_DIR}/prefmask_{other}.json"
        if os.path.exists(mpath):
            with open(mpath) as f:
                other_mask = json.load(f)
            disagree = sum(1 for a, b in zip(rm_prefers_human, other_mask) if a != b)
            mask = [a != b for a, b in zip(rm_prefers_human, other_mask)]
            with open(f"{OUT_DIR}/disagree_{args.labeler}_vs_{other}.json", "w") as f:
                json.dump(mask, f)
            print(f"  disagreement {args.labeler} vs {other}: {disagree}/{n} "
                  f"({disagree/n*100:.2f}%)  (mask saved — where the causal action is)")


if __name__ == "__main__":
    main()
