# Framing-bias RM scoring — harmless-only vs mixed-pool comparison

With-prompts mode only (recovered real prompts; all 50 pairs matched for every checkpoint). Full-sample cut shown below (error-excluded and length-parity cuts are identical to full-sample for structural_formatting — no error-flagged or length-excluded pairs fall in that dimension; see per-seed summaries in results/framing_bias_pair_scores_mixed_seed{42,0,1}.json for the other cuts).

## Win rate toward the (+) pole, by dimension

| dimension / pooled | harmless_seed42 (old) | mixed_seed42 | mixed_seed0 | mixed_seed1 |
|---|---|---|---|---|
| hedging | 0.353 (n=17) | 0.059 (n=17) | 0.000 (n=17) | 0.059 (n=17) |
| confidence_markers | 0.545 (n=11) | 0.273 (n=11) | 0.182 (n=11) | 0.091 (n=11) |
| structural_formatting | 0.769 (n=13) | 0.538 (n=13) | 0.231 (n=13) | 0.385 (n=13) |
| directness | 0.667 (n=9) | 0.444 (n=9) | 0.222 (n=9) | 0.000 (n=9) |
| **pooled** | 0.560 (n=50) | 0.300 (n=50) | 0.140 (n=50) | 0.140 (n=50) |

## Significance (sign-test p, Wilcoxon p)

| dimension / pooled | harmless_seed42 (old) | mixed_seed42 | mixed_seed0 | mixed_seed1 |
|---|---|---|---|---|
| hedging | sign=0.332, Wilcoxon=0.562 | sign=0.000275, Wilcoxon=0.000153 | sign=1.53e-05, Wilcoxon=3.05e-05 | sign=0.000275, Wilcoxon=6.1e-05 |
| confidence_markers | sign=1, Wilcoxon=0.898 | sign=0.227, Wilcoxon=0.365 | sign=0.0654, Wilcoxon=0.0537 | sign=0.0117, Wilcoxon=0.00195 |
| structural_formatting | sign=0.0923, Wilcoxon=0.0574 | sign=1, Wilcoxon=0.542 | sign=0.0923, Wilcoxon=0.0327 | sign=0.581, Wilcoxon=0.146 |
| directness | sign=0.508, Wilcoxon=0.25 | sign=1, Wilcoxon=0.547 | sign=0.18, Wilcoxon=0.109 | sign=0.00391, Wilcoxon=0.00781 |
| **pooled** | sign=0.48, Wilcoxon=0.243 | sign=0.0066, Wilcoxon=0.0116 | sign=2.1e-07, Wilcoxon=5.39e-08 | sign=2.1e-07, Wilcoxon=7.49e-08 |

## Mean signed delta (RM-score units)

| dimension / pooled | harmless_seed42 (old) | mixed_seed42 | mixed_seed0 | mixed_seed1 |
|---|---|---|---|---|
| hedging | -0.0208 | -0.1922 | -0.1237 | -0.1142 |
| confidence_markers | 0.0509 | -0.0679 | -0.0663 | -0.1347 |
| structural_formatting | 0.0487 | -0.1112 | -0.2555 | -0.2320 |
| directness | 0.0951 | -0.0661 | -0.1200 | -0.1610 |
| **pooled** | 0.0339 | -0.1211 | -0.1447 | -0.1577 |

## structural_formatting — decisive length-residual test

Regression of signed_delta on (reworded_words − original_words), restricted to the 13 structural_formatting pairs, per checkpoint. If the coefficient is near zero and the win-rate/significance pattern is unchanged from the pooled-corpus numbers above, the list-preference effect is separable from length. If the coefficient tracks the checkpoint's overall length bias (harmless: r≈-0.40 pooled; mixed: r≈+0.27 to +0.35 pooled, see results/framing_bias_summary_mixed.md training-comparison context), the structural_formatting effect is likely length pathology in a formatting costume.

| checkpoint | length-residual coef | win rate (+) | mean signed delta | sign-test p | Wilcoxon p | n |
|---|---|---|---|---|---|---|
| harmless_seed42 (old) | 0.00897 | 0.769 | 0.0487 | 0.0923 | 0.0574 | 13 |
| mixed_seed42 | 0.01438 | 0.538 | -0.1112 | 1 | 0.542 | 13 |
| mixed_seed0 | -0.02782 | 0.231 | -0.2555 | 0.0923 | 0.0327 | 13 |
| mixed_seed1 | -0.03093 | 0.385 | -0.2320 | 0.581 | 0.146 | 13 |
