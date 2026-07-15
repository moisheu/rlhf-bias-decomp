# SHP upstream position-bias probe (timestamp-based)

Phase 2 position-bias measurement, replacing the retired RM-swap-test (Test #6): SHP's `labels` field is deterministically `sign(score_A - score_B)`, so annotator-side A/B slot bias is structurally impossible here. This was designed to instead probe an upstream, data-generating-process bias — the Reddit visibility effect, where earlier-posted comments accumulate more views/upvotes over time — using `seconds_difference`, the same variable the original SHP paper uses to control for this. **The headline result is that SHP's construction already eliminates this exact confound by design (see below), so the planned regression is not estimable in the way originally framed. That null result is itself the finding.**

## Headline finding: SHP filters this confound out by construction

The SHP dataset card states the construction rule directly: pairs are retained only if the *later*-written comment has the higher score, specifically because a higher score for the *earlier* comment would be ambiguous (could reflect quality, or could just reflect more time to accumulate votes). Quoting the card: “If A had been written before B, then we could not conclude \[a higher-scoring A is preferred], since its higher score could have been the result of more visibility.”

Checked over the **entire `train` split** (n=348718, not just the analysis sample): the earlier-posted comment wins **41** pairs total, and every one of those 41 is a case where `created_at_utc_A == created_at_utc_B` (n=85 such exact-timestamp ties in the split; `earlier_side` is an arbitrary tie-break in that case, not a real time ordering). Excluding ties, earlier_won = True in **0 / 348633** pairs with a genuine nonzero time gap — i.e. **0.000%, no exceptions**. The later-posted comment wins every single non-tied pair in the released dataset.

This means SHP is not a usable testbed for measuring naturally occurring Reddit visibility bias: the bias isn't present with some residual rate to estimate — it's been filtered to exactly zero by the dataset's own construction rule. Any regression of `earlier_won` on covariates has an outcome with (essentially) no variance, which is why the logistic regression below fails to converge (perfect separation) rather than returning a small or null coefficient.

**Analysis sample:** SHP `train` split, n=100000 pairs (shuffled, seed=42), used for the tables and regression below (full-split numbers are reported above for the headline claim).

**Sanity check:** `seconds_difference` matches `|created_at_utc_B - created_at_utc_A|` for 100000/100000 rows (max abs diff: 0).

## Overall rate (analysis sample)

Earlier-posted comment wins **0.0001** of pairs (95% CI [0.0001, 0.0002]), n=100000. Chance is 0.5; the ~0 rate here is the construction artifact described above, not a measurement of a small residual effect.

## Stratified by |score_A - score_B| quintile

Shown for completeness/transparency: since earlier_won is pinned to ~0 by construction, every quintile is expected to show a near-zero rate rather than a fading pattern.

| bin | n | earlier_won rate | 95% CI |
|---|---|---|---|
| (0.999, 2.0] | 25035 | 0.0000 | [0.0000, 0.0002] |
| (2.0, 5.0] | 17936 | 0.0001 | [0.0000, 0.0004] |
| (5.0, 12.0] | 18257 | 0.0001 | [0.0000, 0.0004] |
| (12.0, 38.0] | 18935 | 0.0001 | [0.0000, 0.0003] |
| (38.0, 39626.0] | 19837 | 0.0003 | [0.0001, 0.0006] |

## Stratified by seconds_difference quintile

Same caveat: no quintile has room to show a visibility-driven rise, since the outcome is ~0 everywhere by construction.

| bin | n | earlier_won rate | 95% CI |
|---|---|---|---|
| (-0.001, 1406.0] | 20001 | 0.0005 | [0.0003, 0.0009] |
| (1406.0, 3750.0] | 20008 | 0.0000 | [0.0000, 0.0002] |
| (3750.0, 7665.0] | 19994 | 0.0000 | [0.0000, 0.0002] |
| (7665.0, 15558.2] | 19997 | 0.0000 | [0.0000, 0.0002] |
| (15558.2, 99152798.0] | 20000 | 0.0000 | [0.0000, 0.0002] |

## Stratified by domain (subreddit)

| domain | n | earlier_won rate | 95% CI |
|---|---|---|---|
| askengineers_train | 16447 | 0.0000 | [0.0000, 0.0002] |
| askculinary_train | 13105 | 0.0002 | [0.0000, 0.0006] |
| askbaking_train | 12655 | 0.0000 | [0.0000, 0.0003] |
| changemyview_train | 11017 | 0.0000 | [0.0000, 0.0003] |
| askacademia_train | 8883 | 0.0001 | [0.0000, 0.0006] |
| asksciencefiction_train | 8384 | 0.0000 | [0.0000, 0.0005] |
| legaladvice_train | 5976 | 0.0000 | [0.0000, 0.0006] |
| explainlikeimfive_train | 5788 | 0.0005 | [0.0002, 0.0015] |
| askscience_train | 3795 | 0.0003 | [0.0000, 0.0015] |
| askphilosophy_train | 2888 | 0.0003 | [0.0001, 0.0020] |
| askhr_train | 2383 | 0.0004 | [0.0001, 0.0024] |
| askphysics_train | 2186 | 0.0000 | [0.0000, 0.0018] |
| askdocs_train | 1767 | 0.0000 | [0.0000, 0.0022] |
| askanthropology_train | 1112 | 0.0000 | [0.0000, 0.0034] |
| askhistorians_train | 955 | 0.0000 | [0.0000, 0.0040] |
| askvet_train | 946 | 0.0000 | [0.0000, 0.0040] |
| askcarguys_train | 924 | 0.0011 | [0.0002, 0.0061] |
| asksocialscience_train | 789 | 0.0000 | [0.0000, 0.0048] |

## Logistic regression

`earlier_won ~ log_score_gap + log_seconds_diff`, log_score_gap = log(|score_A - score_B| + 1), log_seconds_diff = log(seconds_difference + 1).

**Not estimable.** MLE fitting failed: `MLE did not converge (quasi-complete separation: outcome has ~no variance)`. This is the expected consequence of the headline finding above — `earlier_won` has (essentially) zero variance in the sample, so there is no separating hyperplane for the logistic model to converge to (perfect/quasi-complete separation), not a software error.

**With domain fixed effects: also not estimable.** `LinAlgError: Singular matrix`

## Interpretation

- The regression could not be fit, so there is no intercept or `log_seconds_diff` coefficient to report. The overall-rate and full-corpus numbers above (0.000% earlier-wins on non-tied pairs) already answer the motivating question directly: the earlier-posted comment does **not** win more often than chance conditional on small score gaps or short time gaps — it essentially never wins at all, by construction.


**Note for the Methods section:** this was designed to measure position bias in the data-generating process (Reddit visibility effects), not in a reward model's judgment, and the headline finding is a fact about SHP's construction rather than about Reddit voting behavior in general — the visibility effect the SHP authors describe is presumably real in raw, unfiltered Reddit data, just not observable in the released, pre-filtered SHP pairs. RM-side position bias would need a separate probe against the framing-pair-scoring harness, once the retrained baseline is in.
