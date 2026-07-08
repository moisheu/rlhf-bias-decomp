"""
Method 1 weight computation: subset-conditional length symmetrization
(spec Method 1 + Refinement 1 thin-cell check).

For the tagged seed=42 mixed pool:
  - dl_i = tok_len(chosen_i) - tok_len(rejected_i), response token count with
    the DistilBERT tokenizer (spec S2). chosen/rejected share the prompt, so
    the response-only diff equals the full-sequence diff.
  - Within each subset, bin |dl| (dl != 0) into K quantile bins; each pair lands
    in cell (subset, bin, sign(dl)). dl == 0 -> weight 1.0.
  - w_i = (n_plus + n_minus) / (2 * n_{sign}) within each (subset, bin) cell, so
    the two signs carry equal weighted mass in every magnitude bin of every
    subset. Cap at 10.0, renormalize to mean 1.0.

REFINEMENT 1 (mandatory thin-cell check, gates the weights):
  - Report the full (subset x bin x sign) cell-count table at K=5.
  - If ANY cell in a subset has < 20 pairs, drop THAT subset to K=3 and
    recompute its bins. Report which subsets ended at K=3 vs K=5.
  - This runs BEFORE final weights are trusted; do not proceed to training
    until the table is reported and any thin-cell collapse is done.

Run: python -m experiments.decomposition.compute_weights
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from src.data_utils import HH_RLHF_SUBSETS, load_hh_rlhf_mixed

POOL_SEED = 42
N_TRAIN = 30000
MODEL_NAME = "distilbert-base-uncased"
TAGS_PATH = "results/mixed_pool_subset_tags_seed42.json"
OUT_PATH = "results/reweight_weights_seed42pool.json"

K_DEFAULT = 5
K_THIN = 3
THIN_CELL_MIN = 20
WEIGHT_CAP = 10.0


def response_of(text: str) -> str:
    return text.rsplit("\n\nAssistant:", 1)[-1].strip()


def compute_dl(pool, tokenizer):
    """Response-token-length gap per pair (spec S2)."""
    dl = np.empty(len(pool), dtype=np.int64)
    chosen = pool["chosen"]
    rejected = pool["rejected"]
    for i in range(len(pool)):
        lc = len(tokenizer(response_of(chosen[i]), add_special_tokens=False)["input_ids"])
        lr = len(tokenizer(response_of(rejected[i]), add_special_tokens=False)["input_ids"])
        dl[i] = lc - lr
    return dl


def quantile_bins(absdl_nonzero, k):
    """Return bin edges (interior) for k quantile bins over positive |dl|.

    Uses unique interior quantile edges; if ties collapse edges the effective
    number of bins is < k (reported by the caller via distinct bin ids).
    """
    qs = np.linspace(0, 1, k + 1)[1:-1]  # interior quantiles
    edges = np.unique(np.quantile(absdl_nonzero, qs))
    return edges


def assign_bins(absdl, edges):
    # bin id in [0, len(edges)]; np.searchsorted with 'right' so equal-to-edge
    # goes to the lower bin boundary consistently.
    return np.searchsorted(edges, absdl, side="right")


def build_cell_table(dl, tags, subset, k):
    """Return dict: (bin_id, sign) -> count, plus bin ids present, for a subset."""
    mask = np.array([t == subset for t in tags]) & (dl != 0)
    idx = np.where(mask)[0]
    absdl = np.abs(dl[idx])
    signs = np.sign(dl[idx])
    edges = quantile_bins(absdl, k)
    bins = assign_bins(absdl, edges)
    table = {}
    for b, s in zip(bins, signs):
        table[(int(b), int(s))] = table.get((int(b), int(s)), 0) + 1
    n_zero = int((np.array([t == subset for t in tags]) & (dl == 0)).sum())
    return {
        "edges": edges.tolist(),
        "bin_ids": sorted(set(int(b) for b in bins)),
        "table": table,
        "n_nonzero": len(idx),
        "n_zero": n_zero,
        "idx": idx,
        "absdl": absdl,
        "signs": signs,
        "bins": bins,
    }


def min_cell_count(cell):
    return min(cell["table"].values()) if cell["table"] else 0


def main():
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading mixed pool (n={N_TRAIN}, seed={POOL_SEED}) and tags...")
    pool = load_hh_rlhf_mixed(n=N_TRAIN, seed=POOL_SEED)
    with open(TAGS_PATH) as f:
        tagdata = json.load(f)
    tags = tagdata["tags"]
    assert len(tags) == len(pool), "tag/pool length mismatch"

    print("Computing response-token-length gaps dl (this tokenizes 60k responses)...")
    dl = compute_dl(pool, tokenizer)
    print(f"  dl: mean={dl.mean():.3f} median={np.median(dl):.1f} "
          f"frac_chosen_longer_raw={(dl>0).mean():.4f} frac_zero={(dl==0).mean():.4f}")

    # --- Refinement 1: thin-cell check at K=5, per subset ---
    print("\n=== Cell-count table (K=5) — Refinement 1 thin-cell check ===")
    subset_k = {}
    cells = {}
    for subset in HH_RLHF_SUBSETS:
        cell5 = build_cell_table(dl, tags, subset, K_DEFAULT)
        mc = min_cell_count(cell5)
        print(f"\n  [{subset}]  nonzero={cell5['n_nonzero']}  zero(dl=0)={cell5['n_zero']}  "
              f"bins_formed={len(cell5['bin_ids'])}  min_cell={mc}")
        print(f"    {'bin':>4} {'|dl| range (approx)':>22} {'n(+)':>7} {'n(-)':>7}")
        edges = cell5["edges"]
        for b in cell5["bin_ids"]:
            lo = "0" if b == 0 else f"{edges[b-1]:.0f}"
            hi = f"{edges[b]:.0f}" if b < len(edges) else "inf"
            nplus = cell5["table"].get((b, 1), 0)
            nminus = cell5["table"].get((b, -1), 0)
            flag = "  <-- THIN" if min(nplus, nminus) < THIN_CELL_MIN else ""
            print(f"    {b:>4} {lo+'..'+hi:>22} {nplus:>7} {nminus:>7}{flag}")

        if mc < THIN_CELL_MIN:
            print(f"    THIN CELL (<{THIN_CELL_MIN}) -> dropping {subset} to K={K_THIN}")
            subset_k[subset] = K_THIN
            cells[subset] = build_cell_table(dl, tags, subset, K_THIN)
            mc3 = min_cell_count(cells[subset])
            print(f"    recomputed at K={K_THIN}: bins_formed={len(cells[subset]['bin_ids'])} min_cell={mc3}")
        else:
            subset_k[subset] = K_DEFAULT
            cells[subset] = cell5

    print("\n  Per-subset K after thin-cell check:")
    for s in HH_RLHF_SUBSETS:
        print(f"    {s:28} K={subset_k[s]}  (min_cell={min_cell_count(cells[s])})")

    # --- compute weights from the (possibly collapsed) cells ---
    weights = np.ones(len(pool), dtype=np.float64)  # dl==0 -> 1.0 by default
    n_at_cap = 0
    for subset in HH_RLHF_SUBSETS:
        c = cells[subset]
        idx, bins, signs = c["idx"], c["bins"], c["signs"]
        # counts per (bin, sign)
        table = c["table"]
        for j, row in enumerate(idx):
            b = int(bins[j])
            s = int(signs[j])
            n_sign = table[(b, s)]
            n_plus = table.get((b, 1), 0)
            n_minus = table.get((b, -1), 0)
            w = (n_plus + n_minus) / (2.0 * n_sign)
            weights[row] = w

    pre_cap_max = float(weights.max())
    n_at_cap = int((weights > WEIGHT_CAP).sum())
    weights = np.minimum(weights, WEIGHT_CAP)
    weights = weights / weights.mean()  # renormalize to mean 1.0

    # --- diagnostics ---
    nonzero = dl != 0
    w_nonzero = weights[nonzero]
    dl_nonzero = dl[nonzero]
    frac_longer_weighted = float((w_nonzero * (dl_nonzero > 0)).sum() / w_nonzero.sum())
    weighted_mean_dl = float((weights * dl).sum() / weights.sum())

    print("\n=== Weight diagnostics ===")
    print(f"  weight: min={weights.min():.4f} max={weights.max():.4f} std={weights.std():.4f} "
          f"mean={weights.mean():.4f}")
    print(f"  pre-cap max weight: {pre_cap_max:.4f}  count at cap (>{WEIGHT_CAP}): {n_at_cap}")
    print(f"  weighted frac chosen-longer (dl!=0): {frac_longer_weighted*100:.3f}%  (target 50% +/-1)")
    print(f"  weighted mean dl: {weighted_mean_dl:.4f}  (target ~0)")
    print("  per-subset weighted frac chosen-longer (dl!=0):")
    subset_diag = {}
    for subset in HH_RLHF_SUBSETS:
        m = np.array([t == subset for t in tags]) & nonzero
        wsub = weights[m]
        dsub = dl[m]
        fl = float((wsub * (dsub > 0)).sum() / wsub.sum()) if wsub.sum() > 0 else float("nan")
        subset_diag[subset] = fl
        print(f"    {subset:28} {fl*100:6.3f}%   (n_nonzero={int(m.sum())})")

    flag = not (0.48 <= frac_longer_weighted <= 0.52)
    if flag:
        print(f"\nSTOP/FLAG: weighted chosen-longer {frac_longer_weighted*100:.3f}% outside 48-52% "
              f"(spec diagnostic gate).", file=sys.stderr)

    payload = {
        "pool_seed": POOL_SEED,
        "n_train": N_TRAIN,
        "length_def": "response_token_count_distilbert",
        "K_default": K_DEFAULT,
        "K_thin": K_THIN,
        "thin_cell_min": THIN_CELL_MIN,
        "weight_cap": WEIGHT_CAP,
        "subset_k": subset_k,
        "diagnostics": {
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_std": float(weights.std()),
            "pre_cap_max": pre_cap_max,
            "n_at_cap": n_at_cap,
            "weighted_frac_chosen_longer_nonzero": frac_longer_weighted,
            "weighted_mean_dl": weighted_mean_dl,
            "per_subset_weighted_frac_longer": subset_diag,
            "frac_zero_dl": float((dl == 0).mean()),
            "raw_frac_chosen_longer": float((dl > 0).mean()),
        },
        "weights": weights.tolist(),
        "dl": dl.tolist(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved {OUT_PATH}")
    if flag:
        sys.exit(1)


if __name__ == "__main__":
    main()
