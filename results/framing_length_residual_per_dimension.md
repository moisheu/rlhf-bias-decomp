# Framing-bias length-residual regression — per dimension

Closes the open thread from `results/framing_bias_summary.md`: that report's
"Length-residual regression coefficient" line is computed **once per cut,
pooled over all 50 (or 45) pairs** — it is not broken down by dimension. This
note reruns `signed_delta ~ word_count_diff` separately for each of the four
framing dimensions (`hedging`, `confidence_markers`, `structural_formatting`,
`directness`), using `scipy.stats.linregress`.

**Method, matched to `experiments/framing/score_framing_pairs.py::length_residual_coefficient`:**
`word_count_diff = reworded_words - original_words` (raw, **not** recoded onto
each dimension's `(+)` axis — same convention the existing pooled coefficient
uses). `signed_delta` *is* recoded onto the `(+)` axis. Reproduced the pooled
coefficients exactly as a sanity check (e.g. empty-prompt/error-excluded
pooled slope = -0.00017, matching `framing_bias_summary.md` line 52) before
trusting the per-dimension breakdown.

Caveat worth flagging for `structural_formatting` specifically: 10 of its 13
pairs are `prose_to_list` reword direction and 3 are `list_to_prose`, so raw
`word_count_diff` is not perfectly aligned with the `(+)`/"list" axis across
the subset — same limitation the original pooled coefficient already carried,
just now visible at the dimension level.

## harmless_seed42 — with-prompts mode, error-excluded cut (45 pairs, drops indices [25, 34, 39, 45, 48])

This is the primary cut for the paper numbers.

| dimension | n | slope | SE | p | R² |
|---|---|---|---|---|---|
| hedging | 15 | 0.02693 | 0.01041 | **0.0225** | 0.340 |
| confidence_markers | 9 | -0.04637 | 0.01638 | **0.0253** | 0.534 |
| **structural_formatting** | **13** | **0.00897** | **0.00680** | **0.2134** | **0.137** |
| directness | 8 | 0.12136 | 0.09427 | 0.2454 | 0.216 |
| pooled | 45 | 0.00885 | 0.00974 | 0.3686 | 0.019 |

## harmless_seed42 — empty-prompt mode, error-excluded cut (45 pairs) — brief comparison

| dimension | n | slope | SE | p | R² |
|---|---|---|---|---|---|
| hedging | 15 | 0.01153 | 0.00882 | 0.2135 | 0.116 |
| confidence_markers | 9 | -0.03792 | 0.01209 | **0.0165** | 0.584 |
| **structural_formatting** | **13** | **0.00135** | **0.00560** | **0.8135** | **0.005** |
| directness | 8 | 0.06169 | 0.06550 | 0.3827 | 0.129 |
| pooled | 45 | -0.00017 | 0.00702 | 0.9807 | 0.00001 |

## Four-checkpoint per-dimension comparison (with-prompts mode, full sample)

Full-sample cut shown (matches the style of `framing_bias_summary_mixed.md`
Chunk 10). Error-excluded is identical to full-sample for
`structural_formatting` on every checkpoint — none of the 5 error-flagged
pairs fall in that dimension — confirmed by rerunning both cuts.

| dimension | harmless_seed42 (old) | mixed_seed42 | mixed_seed0 | mixed_seed1 |
|---|---|---|---|---|
| hedging | n=17, slope=0.02324, se=0.00957, p=0.0282, R²=0.282 | n=17, slope=0.00220, se=0.01043, p=0.8357, R²=0.003 | n=17, slope=0.00047, se=0.00326, p=0.8861, R²=0.001 | n=17, slope=0.00180, se=0.00314, p=0.5739, R²=0.022 |
| confidence_markers | n=11, slope=-0.04379, se=0.01565, p=0.0208, R²=0.465 | n=11, slope=-0.03163, se=0.01476, p=0.0608, R²=0.338 | n=11, slope=-0.00372, se=0.00699, p=0.6079, R²=0.030 | n=11, slope=-0.00314, se=0.00911, p=0.7386, R²=0.013 |
| **structural_formatting** | **n=13, slope=0.00897, se=0.00680, p=0.2134, R²=0.137** | **n=13, slope=0.01438, se=0.03622, p=0.6989, R²=0.014** | **n=13, slope=-0.02782, se=0.02278, p=0.2477, R²=0.119** | **n=13, slope=-0.03093, se=0.02813, p=0.2950, R²=0.099** |
| directness | n=9, slope=0.07418, se=0.07569, p=0.3597, R²=0.121 | n=9, slope=0.04478, se=0.03680, p=0.2630, R²=0.175 | n=9, slope=0.00668, se=0.03054, p=0.8332, R²=0.007 | n=9, slope=0.00276, se=0.02706, p=0.9215, R²=0.002 |
| pooled | n=50, slope=0.00630, se=0.00882, p=0.4789, R²=0.011 | n=50, slope=-0.00243, se=0.00979, p=0.8047, R²=0.001 | n=50, slope=-0.00230, se=0.00615, p=0.7103, R²=0.003 | n=50, slope=-0.00249, se=0.00684, p=0.7175, R²=0.003 |

## Interpretation

**structural_formatting: no evidence of length-mediation, at any checkpoint — but this is a low-power, not a clean, result.**

Across all four checkpoints (harmless_seed42, mixed_seed42, mixed_seed0,
mixed_seed1) and both the with-prompts and empty-prompt scoring modes, the
`structural_formatting` length-residual coefficient is **never significant**
(p ranges 0.21–0.81) and its **sign is not stable** — positive at
harmless_seed42 and mixed_seed42, negative at mixed_seed0 and mixed_seed1.
If the dimension's positive win-rate effect (e.g. 0.769 at harmless_seed42
with-prompts, `framing_bias_summary.md` line 76) were really just "the RM
rewards longer responses and list-formatted rewords happen to be longer,"
you'd expect a consistent, non-trivial slope. That's not what shows up here.

So: **the effect survives length control in the narrow sense that no
significant length confound is detected** — it does not collapse into a
length effect. But at n=13 per checkpoint, R² up to 0.137 without reaching
significance means the test is underpowered to rule out a modest
length-mediated component either. This settles the question at the
confidence level the pair count actually supports: no positive evidence for
"list preference is length in disguise," not a proof that length plays no
role.

**Side finding surfaced only by going per-dimension:** `hedging` and
`confidence_markers` *do* show significant length-residual coefficients in
several cuts (e.g. hedging p=0.0225–0.0282, confidence_markers
p=0.0165–0.0253 at harmless_seed42). The pooled-only coefficient in
`framing_bias_summary.md` was masking this — those two dimensions'
length-sensitivity was invisible in the pooled number. Worth a note if either
dimension gets cited for a length-independent effect elsewhere in the
writeup.

## Scoring script check: pooled-only design — oversight, not intentional

`experiments/framing/score_framing_pairs.py::build_cut_summary` (lines
194–202):

```python
def build_cut_summary(records: list[dict], key: str = "signed_delta") -> dict:
    by_dim = {}
    for dim in DIMENSION_AXES:
        dim_records = [r for r in records if r["dimension"] == dim]
        if dim_records:
            by_dim[dim] = dimension_stats(dim_records, key)
    pooled = dimension_stats(records, key) if records else {}
    residual_coef = length_residual_coefficient(records, key) if records else None
    return {"by_dimension": by_dim, "pooled": pooled, "length_residual_coefficient": residual_coef}
```

The per-dimension loop (`by_dim`) already exists and calls `dimension_stats`
(win rate, sign-test, Wilcoxon) on `dim_records` for each dimension
individually. `length_residual_coefficient` is a separate, already-written,
general-purpose function — it takes any `records` list and a `key`, with no
assumption baked in about pooling. It is called exactly once, outside that
loop, on the full `records` list.

There's no comment, flag, or docstring anywhere in the file suggesting the
pooled restriction was a deliberate methodological choice (e.g. "n too small
per dimension to regress" — n=8/9/13/15/17 are all ≥3, well above the
function's own `len(xs) < 2` guard). Given the per-dimension loop for the
other three statistics was already right there, the natural reading is that
`length_residual_coefficient` was written and tested against the pooled case
first and never wired into the `by_dim` loop alongside `dimension_stats` —
**an oversight**, not intentional pooled-only design. Recommend adding it to
`by_dim` in a follow-up so future runs (new seeds, mixed-pool checkpoints)
get the per-dimension breakdown by default instead of needing an ad hoc
rerun like this one.
