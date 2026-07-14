"""
Phase 4 decision-table summary for the failed-Phase-3 pivot (both arms).

  Arm A: Policy-RAW vs Policy-HUMAN  — does RM relabeling per se inject length
         bias relative to human labels?
  Arm B: Policy-RAW vs Policy-REWEIGHT — does a ~9% RM-level bias reduction
         change downstream behavior, or is 9% below the propagation threshold?

Both arms are scored against the pre-registered decision table VERBATIM, with the
"CORR" role played by HUMAN (Arm A) and REWEIGHT (Arm B).

Reads results/phase4/gen_{raw,human,reweight}_seed{42,0}.json and gen_sft.json.
Run: python -m experiments.dpo.summarize_phase4
"""
import json
import os
import statistics

from scipy.stats import wilcoxon

GEN = "results/phase4"
SEEDS = [42, 0]
REP_GAP_LIMIT = 0.15   # distinct-4-gram arm gap (15 points) -> contamination
TRUNC_LIMIT = 0.20


def load(label):
    p = f"{GEN}/gen_{label}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def lens(rec):
    return [r["gen_len"] for r in rec["records"]]


def mean_distinct4(rec):
    return statistics.mean(r["distinct4"] for r in rec["records"])


REP_KEEP_THRESHOLD = 0.5  # keep generations with distinct-4-gram rate >= this


def paired(raw_rec, other_rec, rep_filter=False):
    """Paired length comparison RAW vs OTHER (aligned by prompt index).

    rep_filter=True drops any prompt where EITHER policy's generation is >50%
    repeated 4-grams (distinct4 < 0.5) -- the spec's repetition-robustness cut,
    to test whether a length difference survives once degeneration loops are
    removed (the failure mode where GPT-2-124M inflates length via repetition).
    """
    rr = raw_rec["records"]; orr = other_rec["records"]
    n = min(len(rr), len(orr))
    lr, lo = [], []
    for i in range(n):
        if rep_filter and (rr[i]["distinct4"] < REP_KEEP_THRESHOLD
                           or orr[i]["distinct4"] < REP_KEEP_THRESHOLD):
            continue
        lr.append(rr[i]["gen_len"]); lo.append(orr[i]["gen_len"])
    kept = len(lr)
    if kept == 0:
        return {"n": 0, "kept": 0}
    diffs = [lr[i] - lo[i] for i in range(kept)]
    nz = [d for d in diffs if d != 0]
    try:
        p = wilcoxon(nz).pvalue if nz else float("nan")
    except ValueError:
        p = float("nan")
    return {
        "n": n, "kept": kept,
        "raw_median": statistics.median(lr), "other_median": statistics.median(lo),
        "raw_mean": statistics.mean(lr), "other_mean": statistics.mean(lo),
        "mean_diff": statistics.mean(diffs), "wilcoxon_p": p,
        "frac_raw_longer": sum(d > 0 for d in diffs) / kept,
    }


def policy_diagnostics():
    """Per-policy quality table: length, truncation, repetition."""
    print("\n  Per-policy diagnostics:")
    print(f"    {'policy':16} {'n':>4} {'mean_len':>9} {'med_len':>8} "
          f"{'trunc%':>7} {'distinct4':>10}")
    for lbl in ["sft"] + [f"{k}_seed{s}" for k in ("raw", "human", "reweight") for s in SEEDS]:
        rec = load(lbl)
        if not rec:
            continue
        ln = lens(rec)
        trunc = rec.get("truncation_rate", 0) * 100
        d4 = mean_distinct4(rec)
        print(f"    {lbl:16} {len(ln):>4} {statistics.mean(ln):>9.1f} "
              f"{statistics.median(ln):>8.0f} {trunc:>6.1f}% {d4:>10.3f}")


def verdict(arm, per_seed, raw_recs, other_recs, sft_rec):
    """Apply the pre-registered table. `other` plays the CORR role."""
    # direction agreement across seeds (sign of mean_diff, RAW - OTHER)
    dirs = [1 if per_seed[s]["mean_diff"] > 0 else -1 for s in SEEDS if s in per_seed]
    same_dir = len(set(dirs)) == 1 and len(dirs) == len(SEEDS)
    sig = all(per_seed[s]["wilcoxon_p"] < 0.05 for s in SEEDS if s in per_seed) and len(dirs) == len(SEEDS)
    raw_longer = same_dir and dirs[0] == 1

    # repetition contamination: |mean distinct4 gap| across arms, any seed
    rep_bad = False
    for s in SEEDS:
        if s in raw_recs and s in other_recs:
            if abs(mean_distinct4(raw_recs[s]) - mean_distinct4(other_recs[s])) > REP_GAP_LIMIT:
                rep_bad = True

    # movement from SFT: did BOTH arms move? (mean length shift)
    moved = None
    if sft_rec:
        sft_mean = statistics.mean(lens(sft_rec))
        raw_moved = any(abs(statistics.mean(lens(raw_recs[s])) - sft_mean) > 3 for s in raw_recs)
        other_moved = any(abs(statistics.mean(lens(other_recs[s])) - sft_mean) > 3 for s in other_recs)
        moved = raw_moved and other_moved

    # truncation
    trunc_bad = any(r.get("truncation_rate", 0) > TRUNC_LIMIT
                    for r in list(raw_recs.values()) + list(other_recs.values()))

    if not same_dir:
        return "INCONCLUSIVE (seeds disagree in direction)"
    if rep_bad:
        return "INCONCLUSIVE (repetition contamination: distinct-4-gram arm gap > 15 pts)"
    if moved is False:
        return "INCONCLUSIVE (neither/an arm did not move from SFT)"
    if trunc_bad:
        return "INCONCLUSIVE (truncation > 20% uncorrected)"
    if raw_longer and sig:
        # (CORR collapse check is reported separately via cross-scores)
        return "BIAS PROPAGATED + correction/labels changed downstream behavior"
    if moved and not sig:
        return "NO DOWNSTREAM TRANSFER (informative null: bias at this magnitude does not survive DPO at this scale)"
    return "INCONCLUSIVE (does not meet any clean criterion — inspect numbers)"


def cross_score_table(recs_by_policy):
    print("\n  Cross-scoring (mean RM score; rows=policy, cols=scoring RM):")
    print(f"    {'policy':16} {'under raw-RM':>14} {'under reweight-RM':>18}")
    for label, rec in recs_by_policy.items():
        if not rec:
            continue
        rr = [r.get("rm_raw_score") for r in rec["records"] if r.get("rm_raw_score") is not None]
        rw = [r.get("rm_reweight_score") for r in rec["records"] if r.get("rm_reweight_score") is not None]
        a = f"{statistics.mean(rr):+.3f}" if rr else "n/a"
        b = f"{statistics.mean(rw):+.3f}" if rw else "n/a"
        print(f"    {label:16} {a:>14} {b:>18}")


def report_arm(name, other_key, question):
    print("\n" + "=" * 78)
    print(f"ARM {name}: Policy-RAW vs Policy-{other_key.upper()}")
    print(f"  Q: {question}")
    print("=" * 78)
    raw_recs = {s: load(f"raw_seed{s}") for s in SEEDS}
    other_recs = {s: load(f"{other_key}_seed{s}") for s in SEEDS}
    sft_rec = load("sft")
    per_seed = {}
    for s in SEEDS:
        if raw_recs[s] and other_recs[s]:
            per_seed[s] = paired(raw_recs[s], other_recs[s])
            r = per_seed[s]
            print(f"  seed {s}: RAW median {r['raw_median']:.0f} vs {other_key} median "
                  f"{r['other_median']:.0f}  (mean diff {r['mean_diff']:+.1f} tok, "
                  f"Wilcoxon p={r['wilcoxon_p']:.4g}, RAW-longer {r['frac_raw_longer']*100:.0f}%)")
        else:
            print(f"  seed {s}: missing generations (raw={bool(raw_recs[s])}, {other_key}={bool(other_recs[s])})")
    # Repetition-robustness cut (spec): does the effect survive dropping
    # generations that are >50% repeated 4-grams (degeneration loops)?
    print("  -- repetition-robust cut (drop >50%-repeated-4gram generations) --")
    for s in SEEDS:
        if raw_recs[s] and other_recs[s]:
            rc = paired(raw_recs[s], other_recs[s], rep_filter=True)
            if rc.get("kept", 0) == 0:
                print(f"  seed {s}: no generations survive the cut")
                continue
            print(f"  seed {s}: kept {rc['kept']}/{rc['n']}  "
                  f"RAW median {rc['raw_median']:.0f} vs {other_key} {rc['other_median']:.0f}  "
                  f"(mean diff {rc['mean_diff']:+.1f} tok, Wilcoxon p={rc['wilcoxon_p']:.4g}, "
                  f"RAW-longer {rc['frac_raw_longer']*100:.0f}%)")

    if len([s for s in SEEDS if s in per_seed]) == len(SEEDS):
        v = verdict(name, per_seed, {s: raw_recs[s] for s in SEEDS if raw_recs[s]},
                    {s: other_recs[s] for s in SEEDS if other_recs[s]}, sft_rec)
        print(f"\n  VERDICT (Arm {name}, pre-registered): {v}")
    else:
        print(f"\n  VERDICT (Arm {name}): PENDING (need both seeds)")


def main():
    print("PHASE 4 — DOWNSTREAM DPO BIAS-PROPAGATION (failed-Phase-3 pivot, both arms)")
    sft = load("sft")
    if sft:
        print(f"SFT baseline mean gen_len: {statistics.mean(lens(sft)):.1f}")
    policy_diagnostics()
    report_arm("A", "human",
               "Does RM-level length bias per se transfer to policy behavior (vs human labels)?")
    report_arm("B", "reweight",
               "Does a ~9% RM-level bias reduction cause downstream change, or is 9% below threshold?")

    # cross-score sanity across everything available
    all_pol = {}
    for lbl in ["sft"] + [f"{k}_seed{s}" for k in ("raw", "human", "reweight") for s in SEEDS]:
        all_pol[lbl] = load(lbl)
    cross_score_table(all_pol)


if __name__ == "__main__":
    main()
