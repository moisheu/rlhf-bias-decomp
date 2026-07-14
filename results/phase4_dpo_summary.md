# Phase 4 — Downstream DPO Bias-Propagation: Results Summary

**Status:** complete. Both pivot arms land **INCONCLUSIVE** against the pre-registered
decision table — but informatively so. Phase 4 was scoped as light/expendable (Phases 2–3
carry the paper); this is an honest downstream result consistent with the spec's stated
GPT-2 quality-floor limitation.

## Design (as run)

Phase 3's correction methods all failed their pre-registered decision table (substantive,
mechanically verified — see the reweighting-arm diagnostics). Per the spec's anticipated
pivot, Phase 4 ran **both** downstream comparisons rather than choosing one:

- **Arm A — Policy-RAW vs Policy-HUMAN:** does RM-level length bias *per se* transfer to
  policy behavior, relative to the data's native human labels?
- **Arm B — Policy-RAW vs Policy-REWEIGHT:** does a ~9% RM-level bias reduction (the best
  reweight seed, pooled length-r ≈ +0.289 at seed 0) cause any downstream behavioral
  change, or is 9% below the propagation threshold?

All three policies were DPO-trained from **one shared GPT-2 (124M) SFT checkpoint** (the
only way to keep arm differences un-confounded by initialization). DPO data: 5,000 pairs
from the mixed pool, disjoint from the SFT 5k, relabeled by each RM (raw mixed / reweight /
human). β=0.1 fixed. Seeds 42 and 0 per arm. Eval: 200 prompts (seed 42) from the
937-filtered eval set; generation temp 0.7 / top-p 0.95 / max_new_tokens 512; paired
Wilcoxon on per-prompt length differences.

## Methodological note — an EOS-masking bug found and fixed mid-run

An initial full run showed **every policy's median generation at exactly the token cap**
(256, then 512) — pure truncation, masking all real length differences. Root cause
(confirmed by reading the transformers 5.9.0 source, not guessed): `DataCollatorFor­
LanguageModeling(mlm=False)` masks every token equal to `pad_token_id` out of the loss.
GPT-2 has no dedicated pad token, so the standard `pad_token = eos_token` made pad and EOS
share an ID — silently erasing the loss signal for the one real end-of-response EOS in
every SFT example. The model therefore never learned to stop, and no token cap could fix
it. Fix: a custom collator that keeps the *first* EOS per sequence as a live target and
masks only the padding EOS after it; EOS also appended to DPO chosen/rejected completions.
Verified end-to-end on CPU before spending GPU (SFT and DPO both went 0/8 → 8/8 on natural
stopping). This is a general GPT-2-SFT gotcha worth flagging in the paper's methods.

## Day-1 relabel signal (Arm A mechanism, pre-DPO)

The raw mixed RM, used as a relabeler, agrees with human labels on **71.8%** of the 5k
pairs (flips 1,411). In flipping them it **injects a length preference**: human labels are
52.1% chosen-longer (mean +19 char gap); after raw-RM relabeling, 57.6% chosen-longer
(+51 char gap). So the label-level length bias is real and measurable *before* any DPO —
the downstream question is whether it survives into policy behavior.

## Per-policy generation diagnostics (200 prompts each)

| policy | mean len | median | truncation % | distinct-4-gram |
|---|---|---|---|---|
| sft | 235.6 | 174 | 24.5% | 0.674 |
| raw_seed42 | 294.1 | 258 | 33.5% | 0.534 |
| raw_seed0 | 301.0 | 285 | 36.0% | 0.511 |
| human_seed42 | 266.6 | 222 | 30.5% | 0.635 |
| human_seed0 | 233.6 | 172 | 21.0% | 0.672 |
| reweight_seed42 | 287.7 | 251 | 33.0% | 0.545 |
| reweight_seed0 | 283.9 | 240 | 32.0% | 0.524 |

Truncation is high across the board — **including the SFT baseline (24.5%)** — so it is an
intrinsic GPT-2-124M property (it rambles on ~1/3 of diverse prompts even after learning
EOS), not a fixable cap setting. RAW/REWEIGHT policies are notably more repetitive
(distinct-4-gram ≈ 0.51–0.55) than HUMAN (≈ 0.64–0.67).

## Arm A — Policy-RAW vs Policy-HUMAN

| | RAW median | HUMAN median | mean diff | Wilcoxon p | RAW-longer |
|---|---|---|---|---|---|
| seed 42 (full) | 258 | 222 | +27.5 | 0.047 | 50% |
| seed 0 (full) | 285 | 172 | +67.4 | 5.5e-5 | 59% |
| seed 42 (rep-robust, kept 77/200) | 114 | 100 | +8.0 | 0.29 | 56% |
| seed 0 (rep-robust, kept 65/200) | 104 | 90 | +13.5 | 0.18 | 60% |

**RAW is consistently longer than HUMAN across both seeds (significant in the full sample)
— length bias transfers directionally.** But the repetition-robustness cut is decisive:
only ~35% of generations survive (≈65% are >50% repeated 4-grams), and among those clean
generations the effect drops to +8/+13.5 tokens and loses significance. **The extra length
is largely repetition-driven degeneration, not clean learned verbosity.**

**Pre-registered verdict: INCONCLUSIVE** (repetition contamination: distinct-4-gram arm gap
> 15 pts). Honest reading: directional transfer present, but at 124M scale it cannot be
cleanly separated from degeneration dynamics — exactly the spec's honest-failure-mode #1.

## Arm B — Policy-RAW vs Policy-REWEIGHT

| | RAW median | REWEIGHT median | mean diff | Wilcoxon p |
|---|---|---|---|---|
| seed 42 (full) | 258 | 251 | +6.4 | 0.66 |
| seed 0 (full) | 285 | 240 | +17.1 | 0.28 |
| seed 42 (rep-robust) | 115 | 107 | +12.2 | 0.46 |
| seed 0 (rep-robust) | 100 | 115 | −7.1 | 0.97 |

**No detectable difference between RAW and REWEIGHT** in any cut; after the repetition
filter the seeds even disagree in direction. **A ~9% RM-level bias reduction produces no
measurable downstream behavioral change** — consistent with "9% is below the propagation
threshold at this scale" (with the caveat that the truncation/repetition noise floor here
is high).

**Pre-registered verdict: INCONCLUSIVE** (truncation > 20% uncorrected — intrinsic to
GPT-2-124M, see above).

## Cross-scoring sanity (mean RM score; rows = policy, cols = scoring RM)

| policy | under raw-RM | under reweight-RM |
|---|---|---|
| sft | +0.479 | +0.079 |
| raw_seed42 | +1.264 | +0.463 |
| raw_seed0 | +1.330 | +0.478 |
| human_seed42 | +0.866 | +0.231 |
| human_seed0 | +0.853 | +0.279 |
| reweight_seed42 | +1.242 | +0.546 |
| reweight_seed0 | +1.342 | +0.634 |

DPO worked: every policy scores far above the SFT baseline under its RM. Crucially, the
**reweight policies did not collapse** — they score comparably to (indeed slightly above)
RAW under the reweight-RM. So the correction did not degrade the policy; it simply produced
no length-behavior change.

## Bottom line

1. **RM length bias transfers directionally into policy behavior** (RAW > HUMAN, both
   seeds), but at GPT-2-124M scale the transfer is confounded with repetition-driven
   degeneration and does not survive a repetition-robustness cut. Claim only that "length
   behavior shifted in the predicted direction," per the spec's GPT-2 caveat.
2. **The 9% RM-level correction shows no downstream effect** (Arm B null across all cuts).
3. Both arms are INCONCLUSIVE per the pre-registered table, for the honest reasons above —
   which is itself a legitimate, pre-anticipated Phase-4 outcome. Phases 2–3 carry the
   paper's core claims.

## Limitations (for the paper)

- **GPT-2-124M quality floor:** ~24–36% truncation and heavy repetition even at the SFT
  baseline; length effects are entangled with degeneration. The repetition-robust cut is
  the minimum rigor; we do not claim beyond "length behavior shifted."
- **Relabeler accuracy:** both RMs are ~59–61% accurate; relabeling injects shared noise.
  Conclusions are about *this RM class*, not frontier RMs.
- **Single β (0.1), single bias axis (length), 124M policy** — all scope limits, not flaws.
- We do **not** claim "debiasing improves policy quality" — the licensed claim is about bias
  *transfer* and its (non-)interruption; cross-scores are guardrails only.
