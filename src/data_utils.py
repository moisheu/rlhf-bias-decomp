import random

from datasets import Dataset, concatenate_datasets, load_dataset

HH_RLHF_SUBSETS = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
]


def _drop_extra_columns(ds):
    cols_to_drop = [c for c in ds.column_names if c not in ("chosen", "rejected", "subset")]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)
    return ds


def _load_hh_rlhf_raw(split: str, by_subset: bool):
    """Load HH-RLHF, tagging each example with its source data_dir if by_subset."""
    if not by_subset:
        return load_dataset("Anthropic/hh-rlhf", split=split)

    tagged = []
    for subset in HH_RLHF_SUBSETS:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
        ds = ds.add_column("subset", [subset] * len(ds))
        tagged.append(ds)
    return concatenate_datasets(tagged)


def load_hh_rlhf_subset(n: int = 5000, split: str = "train", by_subset: bool = False):
    """Return an n-example subset of HH-RLHF with only chosen/rejected columns.

    WARNING: this is a non-shuffling prefix loader — it takes ds.select(range(n))
    over HF's default concatenation order. Since "harmless-base" is the first
    42,537-row block in that concatenation, any n <= 42537 (e.g. the n=30000
    training pool) is 100% harmless-base. Do not use this for training; use
    load_hh_rlhf_mixed instead. Kept as-is (unshuffled) for backward
    compatibility with prior runs that need to be reproduced exactly.

    by_subset=True tags each example with its source data_dir (see
    HH_RLHF_SUBSETS) before concatenating, and keeps that "subset" column in
    the output instead of dropping it.
    """
    ds = _load_hh_rlhf_raw(split, by_subset)
    subset = ds.select(range(n))
    return _drop_extra_columns(subset)


def load_hh_rlhf_mixed(n: int = 5000, split: str = "train", seed: int = 42):
    """Return an n-example subset of HH-RLHF, shuffled across all four subsets
    before truncating, so the result is representative of corpus-wide subset
    proportions instead of being dominated by whichever subset is first in
    HF's default concatenation order (see load_hh_rlhf_subset's warning).
    """
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    subset = ds.shuffle(seed=seed).select(range(n))
    return _drop_extra_columns(subset)


def load_hh_rlhf_eval_subset(n: int = 500, split: str = "test", seed: int = 42, by_subset: bool = False):
    """Return a shuffled, seeded n-example held-out subset of HH-RLHF for eval.

    by_subset=True tags each example with its source data_dir (see
    HH_RLHF_SUBSETS) before concatenating, and keeps that "subset" column in
    the output instead of dropping it.
    """
    ds = _load_hh_rlhf_raw(split, by_subset)
    subset = ds.shuffle(seed=seed).select(range(n))
    return _drop_extra_columns(subset)


def _format_hh(instruction: str, response: str) -> str:
    """Render an (instruction, response) pair in HH-RLHF chat format."""
    return f"\n\nHuman: {instruction}\n\nAssistant: {response}"


def load_ultrafeedback_pairs(
    n: int | None = 5000,
    split: str = "train",
    min_gap: float = 1.0,
    seed: int = 42,
    keep_metadata: bool = False,
):
    """Build chosen/rejected pairs from UltraFeedback overall_score ratings.

    Per prompt, every completion pair whose overall_score gap is >= min_gap
    (strictly > 0 if min_gap is 0) is a candidate, and ONE candidate is
    sampled uniformly at random (seeded). min_gap only filters near-ties at
    the judge's noise floor; uniform sampling — rather than best-vs-worst —
    keeps pair selection independent of gap size, since gap size correlates
    with length/style divergence and conditioning on it would contaminate
    downstream bias measurements. One pair per prompt keeps examples
    independent.

    Returns the load_hh_rlhf_subset schema: chosen/rejected strings in
    "\\n\\nHuman: ...\\n\\nAssistant: ..." format. keep_metadata=True adds
    instruction, model names, scores, and score_gap columns (dropped by the
    training pipeline; useful for bias analysis).
    """
    ds = load_dataset("openbmb/UltraFeedback", split=split)
    # UltraFeedback concatenates its source datasets in contiguous blocks;
    # shuffle so a truncated subset is not dominated by a single source.
    ds = ds.shuffle(seed=seed)
    rng = random.Random(seed)
    effective_gap = max(min_gap, 1e-9)

    rows = []
    for ex in ds:
        if n is not None and len(rows) >= n:
            break
        instruction = ex["instruction"]

        completions = []
        for comp in ex["completions"]:
            response = (comp.get("response") or "").strip()
            try:
                score = float(comp["overall_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if response:
                completions.append((response, score, comp.get("model", "")))

        candidates = [
            (a, b) if a[1] > b[1] else (b, a)
            for i, a in enumerate(completions)
            for b in completions[i + 1 :]
            if abs(a[1] - b[1]) >= effective_gap
        ]
        if not candidates:
            continue

        chosen, rejected = rng.choice(candidates)
        row = {
            "chosen": _format_hh(instruction, chosen[0]),
            "rejected": _format_hh(instruction, rejected[0]),
        }
        if keep_metadata:
            row.update(
                instruction=instruction,
                chosen_model=chosen[2],
                rejected_model=rejected[2],
                score_chosen=chosen[1],
                score_rejected=rejected[1],
                score_gap=chosen[1] - rejected[1],
            )
        rows.append(row)

    return Dataset.from_list(rows)


def load_shp_pairs(
    n: int | None = 5000,
    split: str = "train",
    seed: int = 42,
    keep_metadata: bool = False,
):
    """Build chosen/rejected pairs from SHP (Stanford Human Preferences).

    Unlike UltraFeedback, SHP rows are already single A/B pairs with a
    label, so no candidate construction or sampling is needed — labels is 1
    iff score_A > score_B (verified exactly on the cached data: 100%
    consistent, no ties), so it just selects which side is chosen.

    SHP is stored in contiguous per-domain (subreddit) blocks like
    UltraFeedback's source blocks, so shuffle before truncating to n so a
    subset is not dominated by a single domain.

    Returns the load_hh_rlhf_subset schema: chosen/rejected strings in
    "\\n\\nHuman: ...\\n\\nAssistant: ..." format (history is the
    instruction). keep_metadata=True adds human_ref_A, human_ref_B, labels,
    score_A, score_B (dropped by the training pipeline; useful for the
    A-vs-B slot bias analysis).
    """
    ds = load_dataset("stanfordnlp/shp", split=split)
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    rows = []
    for ex in ds:
        a_preferred = ex["labels"] == 1
        chosen = ex["human_ref_A"] if a_preferred else ex["human_ref_B"]
        rejected = ex["human_ref_B"] if a_preferred else ex["human_ref_A"]

        row = {
            "chosen": _format_hh(ex["history"], chosen),
            "rejected": _format_hh(ex["history"], rejected),
        }
        if keep_metadata:
            row.update(
                human_ref_A=ex["human_ref_A"],
                human_ref_B=ex["human_ref_B"],
                labels=ex["labels"],
                score_A=ex["score_A"],
                score_B=ex["score_B"],
            )
        rows.append(row)

    return Dataset.from_list(rows)


def pair_length_stats(ds):
    """Length-gap diagnostics for a chosen/rejected dataset.

    Run this on the constructed pairs to verify the pairing method has not
    amplified the chosen-is-longer correlation beyond what the raw
    annotations carry. (Chosen and rejected share the prompt prefix, so the
    character gap reflects the responses alone.)
    """
    diffs = [len(c) - len(r) for c, r in zip(ds["chosen"], ds["rejected"])]
    n_pairs = len(diffs)
    return {
        "n_pairs": n_pairs,
        "frac_chosen_longer": sum(d > 0 for d in diffs) / n_pairs,
        "mean_char_gap": sum(diffs) / n_pairs,
        "median_char_gap": sorted(diffs)[n_pairs // 2],
    }
