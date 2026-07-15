# Framing-bias reward model scoring — summary

Reward model: `results/reward_model_baseline` (seed 42 baseline).
Pairs: `experiments/framing/framing_pairs.json` (n=50).

Two scoring modes are reported. Empty-prompt mode formats every response as
`\n\nHuman: \n\nAssistant: {text}`, matching how `sample_eval_responses_filtered.py`
stripped prompts before the rewrite step; this is off-distribution for the RM
(trained on real Human/Assistant transcripts), so absolute scores are not
meaningful, but the paired design keeps the within-pair delta valid.
With-prompts mode recovers each response's original prompt by suffix-matching
against the fixed eval subset and rescoring; use these numbers for the paper
if all 50 pairs match.

Deltas are recoded onto each dimension's (+) axis before analysis (see
SCORING_SPEC.md) — a positive signed delta always means the RM rewarded the
pole named in the "win rate (+)" column, regardless of which side of the pair
was the original vs. the reword.

5 pairs (indices [25, 34, 39, 45, 48]) preserve a factual error from
the source response by design (a false burial claim, an inaccurate Monty
Python attribution, a wrong sports record, a wrong historical date and an
unrelated historical anecdote in the same pair, and an erroneous calculation).
The "error-excluded" cut drops these 5; see the robustness comparison per
dimension below. The underlying pair text is not reproduced in this report —
see `results/framing_bias_pair_scores_seed42.json` (scores only) or `experiments/framing/framing_pairs.json` (full text) if needed.

## Empty-prompt mode (sanity check)

### Full sample

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 17 | 0.529 | 0.0009 | 1 | 1 |
| confidence_markers | 11 | 0.364 | 0.0392 | 0.5488 | 0.8984 |
| structural_formatting | 13 | 0.615 | 0.0283 | 0.5811 | 0.2439 |
| directness | 9 | 0.667 | 0.0613 | 0.5078 | 0.5703 |
| **pooled** | 50 | 0.540 | 0.0273 | 0.6718 | 0.4665 |

Length-residual regression coefficient (signed_delta ~ word-count diff): 0.00094

### Error-excluded (dropped 5 pairs: indices [25, 34, 39, 45, 48])

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 15 | 0.533 | 0.0053 | 1 | 0.9341 |
| confidence_markers | 9 | 0.444 | 0.0701 | 1 | 0.8203 |
| structural_formatting | 13 | 0.615 | 0.0283 | 0.5811 | 0.2439 |
| directness | 8 | 0.750 | 0.1080 | 0.2891 | 0.3125 |
| **pooled** | 45 | 0.578 | 0.0432 | 0.3713 | 0.1924 |

Length-residual regression coefficient (signed_delta ~ word-count diff): -0.00017

### Length-parity-filtered (dropped 0 pairs with pct_diff > 10.0%)

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 17 | 0.529 | 0.0009 | 1 | 1 |
| confidence_markers | 11 | 0.364 | 0.0392 | 0.5488 | 0.8984 |
| structural_formatting | 13 | 0.615 | 0.0283 | 0.5811 | 0.2439 |
| directness | 9 | 0.667 | 0.0613 | 0.5078 | 0.5703 |
| **pooled** | 50 | 0.540 | 0.0273 | 0.6718 | 0.4665 |

Length-residual regression coefficient (signed_delta ~ word-count diff): 0.00094

## With-prompts mode (paper numbers)

Prompt recovery: 50/50 pairs matched against the fixed eval subset (n=1000, seed=42).

### Full sample

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 17 | 0.353 | -0.0208 | 0.3323 | 0.5619 |
| confidence_markers | 11 | 0.545 | 0.0509 | 1 | 0.8984 |
| structural_formatting | 13 | 0.769 | 0.0487 | 0.09229 | 0.05737 |
| directness | 9 | 0.667 | 0.0951 | 0.5078 | 0.25 |
| **pooled** | 50 | 0.560 | 0.0339 | 0.4799 | 0.2427 |

Length-residual regression coefficient (signed_delta ~ word-count diff): 0.00630

### Error-excluded (dropped 5 pairs: indices [25, 34, 39, 45, 48])

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 15 | 0.400 | -0.0146 | 0.6072 | 0.9032 |
| confidence_markers | 9 | 0.556 | 0.0709 | 1 | 0.7344 |
| structural_formatting | 13 | 0.769 | 0.0487 | 0.09229 | 0.05737 |
| directness | 8 | 0.625 | 0.0862 | 0.7266 | 0.375 |
| **pooled** | 45 | 0.578 | 0.0387 | 0.3713 | 0.1647 |

Length-residual regression coefficient (signed_delta ~ word-count diff): 0.00885

### Length-parity-filtered (dropped 0 pairs with pct_diff > 10.0%)

| dimension | n | win rate (+) | mean signed delta | sign-test p | Wilcoxon p |
|---|---|---|---|---|---|
| hedging | 17 | 0.353 | -0.0208 | 0.3323 | 0.5619 |
| confidence_markers | 11 | 0.545 | 0.0509 | 1 | 0.8984 |
| structural_formatting | 13 | 0.769 | 0.0487 | 0.09229 | 0.05737 |
| directness | 9 | 0.667 | 0.0951 | 0.5078 | 0.25 |
| **pooled** | 50 | 0.560 | 0.0339 | 0.4799 | 0.2427 |

Length-residual regression coefficient (signed_delta ~ word-count diff): 0.00630
