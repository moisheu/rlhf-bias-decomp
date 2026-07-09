"""
Phase 4 data prep: disjoint SFT and DPO samples from the mixed 30k pool.

The pool `load_hh_rlhf_mixed(n=30000, seed=42)` is already seed-42 shuffled, so
two disjoint contiguous slices give the seed-42 SFT and DPO samples the spec
asks for (DPO disjoint from SFT):
  - SFT set  = pool rows [0:5000]      -> full chosen texts (causal-LM SFT target)
  - DPO set  = pool rows [5000:10000]  -> preference pairs to be relabeled

Prompt / completion split (matches the RM's transcript convention and
score_framing's extract_response): split each side at the FINAL "\\n\\nAssistant:".
The prompt (shared context) is the prefix incl. that marker; the completion is
the response suffix.

Writes:
  results/phase4/sft_texts.json    list[str]  full "Human/Assistant" chosen texts
  results/phase4/dpo_pairs.json    list[dict] {prompt, human_chosen, human_rejected}
                                   (completions only; human_* are the original labels)

Run: python -m experiments.dpo.build_phase4_data
"""
import json
import os

from src.data_utils import load_hh_rlhf_mixed

POOL_SEED = 42
N_POOL = 30000
N_SFT = 5000
N_DPO = 5000
OUT_DIR = "results/phase4"
SEP = "\n\nAssistant:"


def split_prompt_completion(text: str):
    """(prompt incl. final 'Assistant:', completion) for a full transcript."""
    prefix, _, suffix = text.rpartition(SEP)
    if not prefix:  # no Assistant marker (shouldn't happen for HH-RLHF)
        return text, ""
    return prefix + SEP, suffix


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = load_hh_rlhf_mixed(n=N_POOL, seed=POOL_SEED)
    print(f"Loaded pool: {len(pool)} rows")

    sft = pool.select(range(0, N_SFT))
    dpo = pool.select(range(N_SFT, N_SFT + N_DPO))
    print(f"SFT slice: rows [0:{N_SFT}]   DPO slice: rows [{N_SFT}:{N_SFT+N_DPO}] (disjoint)")

    # --- SFT targets: full chosen transcripts ---
    sft_texts = list(sft["chosen"])
    with open(f"{OUT_DIR}/sft_texts.json", "w") as f:
        json.dump(sft_texts, f)
    print(f"Wrote {OUT_DIR}/sft_texts.json ({len(sft_texts)} texts)")

    # --- DPO pairs: prompt + human-labeled completions ---
    pairs = []
    prompt_mismatch = 0
    for chosen, rejected in zip(dpo["chosen"], dpo["rejected"]):
        p_c, comp_c = split_prompt_completion(chosen)
        p_r, comp_r = split_prompt_completion(rejected)
        if p_c != p_r:
            prompt_mismatch += 1  # differing multi-turn context; keep chosen's prompt
        pairs.append({"prompt": p_c, "human_chosen": comp_c, "human_rejected": comp_r})
    with open(f"{OUT_DIR}/dpo_pairs.json", "w") as f:
        json.dump(pairs, f)
    print(f"Wrote {OUT_DIR}/dpo_pairs.json ({len(pairs)} pairs)")
    print(f"  prompt-mismatch pairs (chosen/rejected diverge before final Assistant): {prompt_mismatch}")
    # quick length sanity
    import statistics
    lc = [len(p["human_chosen"]) for p in pairs]
    lr = [len(p["human_rejected"]) for p in pairs]
    print(f"  human chosen char-len median {statistics.median(lc):.0f}, "
          f"rejected median {statistics.median(lr):.0f}")


if __name__ == "__main__":
    main()
