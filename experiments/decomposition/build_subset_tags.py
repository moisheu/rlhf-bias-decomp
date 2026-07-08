"""
S1: reconstruct subset tags for the EXISTING seed=42 mixed pool by exact-match
lookup (spec S1). Does NOT resample -- tags the identical 30k rows the mixed
baseline trained on, so the reweighting comparison is not confounded by a pool
change.

Method:
  1. Load each of the four HH-RLHF subsets separately (data_dir=...).
  2. Build sha1(chosen + "\\x00" + rejected) -> subset. First match wins by the
     canonical order below; cross-subset key collisions are counted.
  3. Load the mixed pool exactly as the baseline did:
     load_hh_rlhf_mixed(n=30000, seed=42).
  4. Map each row to its subset. Report coverage / collisions / unmatched.
     STOP if unmatched > 0.5% (spec failure-mode guard #2).
  5. Save results/mixed_pool_subset_tags_seed42.json (row index -> subset).

Run: python -m experiments.decomposition.build_subset_tags
"""
import hashlib
import json
import sys

from datasets import load_dataset

from src.data_utils import HH_RLHF_SUBSETS, load_hh_rlhf_mixed

POOL_SEED = 42
N_TRAIN = 30000
OUT_PATH = "results/mixed_pool_subset_tags_seed42.json"


def key_of(chosen: str, rejected: str) -> str:
    h = hashlib.sha1()
    h.update(chosen.encode("utf-8"))
    h.update(b"\x00")
    h.update(rejected.encode("utf-8"))
    return h.hexdigest()


def main():
    print("Building sha1(chosen\\x00rejected) -> subset lookup from the four subsets...")
    lookup = {}
    cross_subset_collisions = 0
    within_subset_dupes = 0
    subset_sizes = {}
    for subset in HH_RLHF_SUBSETS:  # canonical order = collision tie-break
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
        subset_sizes[subset] = len(ds)
        for chosen, rejected in zip(ds["chosen"], ds["rejected"]):
            k = key_of(chosen, rejected)
            if k in lookup:
                if lookup[k] != subset:
                    cross_subset_collisions += 1  # keep first (earlier subset)
                else:
                    within_subset_dupes += 1
                continue
            lookup[k] = subset
    print(f"  subset sizes: {subset_sizes}")
    print(f"  unique keys: {len(lookup)}  cross-subset collisions: {cross_subset_collisions}  "
          f"within-subset dupes: {within_subset_dupes}")

    print(f"Loading mixed pool (n={N_TRAIN}, seed={POOL_SEED})...")
    pool = load_hh_rlhf_mixed(n=N_TRAIN, seed=POOL_SEED)

    tags = []
    unmatched = 0
    counts = {s: 0 for s in HH_RLHF_SUBSETS}
    for chosen, rejected in zip(pool["chosen"], pool["rejected"]):
        subset = lookup.get(key_of(chosen, rejected))
        if subset is None:
            unmatched += 1
            tags.append(None)
        else:
            counts[subset] += 1
            tags.append(subset)

    n = len(tags)
    coverage = (n - unmatched) / n
    print("\n=== S1 tag reconstruction report ===")
    print(f"  pool rows: {n}")
    print(f"  matched: {n - unmatched}  ({coverage*100:.3f}% coverage)")
    print(f"  unmatched: {unmatched}  ({unmatched/n*100:.3f}%)")
    print(f"  cross-subset collisions in lookup: {cross_subset_collisions}")
    print("  subset composition of the pool:")
    for s in HH_RLHF_SUBSETS:
        print(f"    {s:28} {counts[s]:6d}  ({counts[s]/n*100:5.2f}%)")

    if unmatched / n > 0.005:
        print(f"\nSTOP: unmatched {unmatched/n*100:.3f}% > 0.5% (spec guard #2).", file=sys.stderr)
        sys.exit(1)

    payload = {
        "pool_seed": POOL_SEED,
        "n_train": N_TRAIN,
        "coverage": coverage,
        "unmatched": unmatched,
        "cross_subset_collisions": cross_subset_collisions,
        "within_subset_dupes": within_subset_dupes,
        "subset_sizes": subset_sizes,
        "pool_composition": counts,
        "tags": tags,  # list indexed by pool row index; None if unmatched
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved {OUT_PATH}")
    if coverage < 0.995:
        print(f"WARNING: coverage {coverage*100:.3f}% < 99.5% target -- FLAG.", file=sys.stderr)


if __name__ == "__main__":
    main()
