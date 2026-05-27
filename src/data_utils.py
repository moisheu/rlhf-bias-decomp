from datasets import load_dataset


def load_hh_rlhf_subset(n: int = 5000, split: str = "train"):
    """Return an n-example subset of HH-RLHF with only chosen/rejected columns."""
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    subset = ds.select(range(n))
    cols_to_drop = [c for c in subset.column_names if c not in ("chosen", "rejected")]
    if cols_to_drop:
        subset = subset.remove_columns(cols_to_drop)
    return subset
