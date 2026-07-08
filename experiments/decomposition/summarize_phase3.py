"""
Summarize Phase 3 arms against the pre-registered decision table.

Reads results/phase3_length_corr.json (produced by eval_length_correlation for
each arm) and prints, per method, the mean/range of pooled length-score r and
accuracy across seeds, plus the pre-registered verdict.

Decision table (spec, verbatim -- not moved after seeing results):
  Worked     : mean|pooled r| <= 0.15 AND mean acc >= 0.575 AND
               seeds agree in direction AND seed r-range < 0.15
  Tradeoff   : r criterion met but acc in [0.52, 0.575)
  Failed     : mean|pooled r| > 0.25
  Inconclusive: seed r-range >= 0.15 or mixed signs across seeds

Framing note (Refinement 2): the |r| <= 0.15 line is the pre-registered target,
read as "~55% reduction from the +0.32 mixed baseline" -- an arbitrary but
fixed line, not a principled constant.

Run: python -m experiments.decomposition.summarize_phase3
"""
import json
import os

CORR = "results/phase3_length_corr.json"
METHODS = ["symloss", "reweight", "combined"]
SEEDS = [42, 0, 1]

# Validated local mixed baseline (reproduces Chunk 9): used as reference if the
# baseline rows are not present in the eval file (e.g. on a fresh Colab box).
BASELINE_REF = {"acc": 0.601, "r_pooled": 0.339, "r_range": "0.294..0.374"}


def load():
    if not os.path.exists(CORR):
        raise FileNotFoundError(CORR)
    with open(CORR) as f:
        rows = json.load(f)
    return {r["label"]: r for r in rows}


def verdict(mean_abs_r, mean_acc, r_range, signs):
    same_sign = len(set(signs)) == 1
    if mean_abs_r > 0.25:
        return "FAILED (|r| within noise of baseline)"
    if r_range >= 0.15 or not same_sign:
        return "INCONCLUSIVE (unstable across seeds / mixed signs)"
    if mean_abs_r <= 0.15 and mean_acc >= 0.575 and same_sign:
        return "WORKED (met pre-registered target)"
    if mean_abs_r <= 0.15 and 0.52 <= mean_acc < 0.575:
        return "BIAS-ACCURACY TRADEOFF (publishable, quantified)"
    return "PARTIAL (r not reduced enough but not within-noise; see numbers)"


def main():
    rows = load()

    # baseline row (from file if present, else the validated local reference)
    base_labels = [f"mixed_seed{s}" for s in SEEDS]
    if all(b in rows for b in base_labels):
        br = [rows[b]["r_pooled"] for b in base_labels]
        ba = [rows[b]["accuracy"] for b in base_labels]
        base_r = sum(br) / len(br)
        base_a = sum(ba) / len(ba)
        base_src = "eval file"
    else:
        base_r, base_a = BASELINE_REF["r_pooled"], BASELINE_REF["acc"]
        base_src = "validated local reference"

    print("=" * 78)
    print("PHASE 3 DECOMPOSITION — RESULTS vs PRE-REGISTERED DECISION TABLE")
    print("=" * 78)
    print(f"\nMixed baseline ({base_src}): mean pooled r = +{base_r:.3f}, "
          f"mean acc = {base_a:.3f}")
    print("Pre-registered target: |pooled r| <= 0.15  (~55% reduction from +0.32)\n")

    header = f"{'arm':10} " + " ".join(f"{'s'+str(s):>18}" for s in SEEDS) + \
             f" {'mean r':>8} {'r-range':>8} {'mean acc':>9}  verdict"
    print(header)
    print("-" * len(header))

    for method in METHODS:
        cells = []
        rs, accs, signs = [], [], []
        for s in SEEDS:
            lab = f"{method}_seed{s}"
            if lab in rows:
                r = rows[lab]["r_pooled"]
                a = rows[lab]["accuracy"]
                rs.append(r); accs.append(a); signs.append(1 if r >= 0 else -1)
                cells.append(f"r={r:+.3f} a={a:.3f}")
            else:
                cells.append("--pending--")
        if rs:
            mean_r = sum(rs) / len(rs)
            mean_a = sum(accs) / len(accs)
            r_range = max(rs) - min(rs)
            v = verdict(abs(mean_r), mean_a, r_range, signs) if len(rs) == len(SEEDS) \
                else f"({len(rs)}/{len(SEEDS)} seeds done)"
            row = f"{method:10} " + " ".join(f"{c:>18}" for c in cells) + \
                  f" {mean_r:+8.3f} {r_range:8.3f} {mean_a:9.3f}  {v}"
        else:
            row = f"{method:10} " + " ".join(f"{c:>18}" for c in cells) + "  (no seeds yet)"
        print(row)

    print("\nReduction vs baseline (mean pooled r):")
    for method in METHODS:
        rs = [rows[f"{method}_seed{s}"]["r_pooled"] for s in SEEDS
              if f"{method}_seed{s}" in rows]
        if len(rs) == len(SEEDS):
            mean_r = sum(rs) / len(rs)
            red = (1 - abs(mean_r) / base_r) * 100
            print(f"  {method:10} mean r = {mean_r:+.3f}  ->  {red:.0f}% reduction")


if __name__ == "__main__":
    main()
