# Phase 4 (Light) — Downstream DPO Bias-Propagation Test

**Claim under test:** does the length bias in a reward model propagate into policy behavior, and does the Phase 3 correction measurably change that downstream behavior?
**Hard scope limit:** ~5 days wall-clock. Anything not needed for the core claim is cut.

---

## Core design decision (read first)

DPO does not consume a reward model — it trains directly on preference pairs. The RM therefore enters as a **relabeler**: score both sides of each pair with an RM, keep the RM's preferred side as "chosen." Train identical DPO runs where the ONLY difference is which RM produced the labels. Any behavioral difference between the resulting policies is attributable to the RMs' differing biases.

**Arms:**
- **Policy-RAW:** DPO on pairs relabeled by the raw mixed-pool RM (pooled length r ≈ +0.32)
- **Policy-CORR:** DPO on pairs relabeled by the Phase 3 corrected RM
- **Policy-HUMAN (cheap anchor, run if time):** DPO on the original human labels — situates both RM arms against the data's native preferences

**Which corrected RM:** the Phase 3 arm that lands "Worked" per the pre-registered decision table. If multiple work, use reweight (the primary method). If none reach "Worked" but a "bias-accuracy tradeoff" arm exists, use it and report the accuracy cost alongside. If all Phase 3 arms fail, Phase 4 pivots: run Policy-RAW vs Policy-HUMAN only, reframed as "does RM relabeling per se inject length bias relative to human labels" — still a publishable downstream result.

---

## Setup (lightest viable)

- **Policy model:** GPT-2 (124M). DistilBERT can't generate; GPT-2 is the smallest standard causal LM with solid TRL DPOTrainer support and fits Colab easily.
- **Shared SFT starting point (required):** one brief SFT pass of GPT-2 on the chosen responses of the mixed 30k pool (1 epoch, ~5k examples is enough) → save as `sft_base/`. **Both DPO arms start from this same checkpoint.** Without a shared start, arm differences are confounded by initialization.
- **DPO data:** 5,000 pairs sampled (seed=42) from the mixed train pool, EXCLUDED from SFT's 5k (disjoint sample). Relabel this same 5k with each RM. Report: agreement rate of each RM with human labels, and raw-vs-corrected RM disagreement rate. Save the disagreement mask — it's where the causal action is.
- **DPO config:** TRL DPOTrainer, beta=0.1 (fixed, no sweep — note as scope limit), 1 epoch, same LR/schedule both arms, max_length/max_prompt_length matching RM-side conventions (512/256).
- **Seeds:** 2 per arm (42, 0). Third seed only if wall-clock allows.
- **Output dirs:** `results/dpo_raw_seed{42,0}/`, `results/dpo_corr_seed{42,0}/`, `results/dpo_human_seed42/` (anchor, 1 seed is fine).

---

## Measurement

**Eval prompt set:** 200 prompts, sampled seed=42 from the 937-filtered eval set (Human-turn side only). Disjoint from all training data by construction (train/test split upstream).

**Generation protocol (identical across all policies):** temperature 0.7, top-p 0.95, fixed generation seed, max_new_tokens=256, one generation per prompt per policy. Report the truncation rate (fraction hitting the 256 cap); if >20%, raise the cap and regenerate (eval-only, cheap) — truncation compresses exactly the differences being measured.

**Primary metric — paired length comparison:**
For each prompt, compare token lengths of Policy-RAW vs Policy-CORR generations.
- Report: mean/median length per policy, per seed
- **Paired test:** Wilcoxon signed-rank on per-prompt length differences (paired design is the powerful test at n=200; distribution means are secondary)
- Report fraction of prompts where RAW's generation is longer

**Secondary metrics (guards + context, all cheap):**
1. **Repetition guard:** distinct-4-gram rate per generation. GPT-2-scale models inflate length via repetition loops; if repetition differs materially between arms, length differences are contaminated — report alongside, and report length stats excluding high-repetition generations (>50% repeated 4-grams) as a robustness line.
2. **Quality proxy (cross-scoring):** score all generations with BOTH RMs (2×2: each policy's outputs under each RM). Purpose: show the corrected arm didn't collapse into degenerate short outputs. This is a sanity check, not a headline claim.
3. **Movement check:** also generate from `sft_base/` on the same 200 prompts. If neither DPO arm moved meaningfully from SFT (lengths ~identical to SFT), the runs were too light to show anything — that's the "inconclusive" trigger, not a null.

---

## Pre-registered decision table

| Outcome | Criteria |
|---|---|
| **Bias propagated + correction changed downstream behavior** | RAW generations longer than CORR, same direction both seeds, paired Wilcoxon p < 0.05, repetition rates comparable, CORR's own-RM scores not collapsed |
| **No downstream transfer (informative null)** | Both arms moved from SFT baseline, but no significant RAW-vs-CORR length difference — reportable as "RM-level bias at this magnitude does not survive DPO at this scale" |
| **Inconclusive** | Seeds disagree in direction; or repetition contamination (arm gap > 15 points in distinct-4-gram); or neither arm moved from SFT; or truncation > 20% uncorrected |

---

## Honest failure modes (put these in the paper's Limitations)

1. **GPT-2 quality floor:** 124M generations are weak; length effects could reflect degeneration dynamics rather than learned preference. The repetition guard + SFT movement check are the minimum rigor; do not claim beyond "length behavior shifted."
2. **Relabeler accuracy:** both RMs are ~59-61% accurate — relabeling injects shared noise. The comparison is internally valid (only the RM differs) but conclusions are about *this RM class*, not frontier RMs. Say so.
3. **Beta sensitivity unexplored:** one beta (0.1). A different beta could scale effects up/down. Scope limit, not a flaw — state it.
4. **Single bias axis:** only length is tested downstream. Framing was near-null at the RM level (Chunk 10), so its absence downstream would be uninformative; don't test it.
5. **Do NOT claim "debiasing improves policy quality."** The licensed claim is about bias *transfer* and its interruption. Quality cross-scores are guardrails only.

---

## Timeline

- **Day 1:** Relabel 5k pairs with both RMs (inference only, fast). SFT GPT-2 → `sft_base/`. Report relabel agreement/disagreement stats.
- **Day 2-3:** 4-5 DPO runs, sequential (each ~1-2 h on Colab GPU).
- **Day 4:** Generate 200 × all policies + SFT baseline; compute all metrics.
- **Day 5:** Decision-table verdicts, results summary → `results/phase4_dpo_summary.md`.

**Kill-switch:** if Day 2's first DPO run shows training instability that isn't fixed by one LR halving, stop and report rather than tuning — Phase 4 is expendable; Phases 2-3 already carry the paper.
