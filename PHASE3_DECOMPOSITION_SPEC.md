# Phase 3 Decomposition — Implementation-Ready Spec

**Purpose:** Two bias-correction methods for the length-bias finding, specified to the level where a coding agent can execute without further design decisions.
**Context:** Mixed-pool baseline (Chunk 9) shows reproducible +0.27 to +0.35 pooled length-score correlation across seeds 42/0/1 at 59-61% accuracy. Harmless-only baseline shows -0.38 to -0.40 at ~48%. Sign of length bias is training-composition dependent. Phase 3 asks: can we remove the length signal without destroying accuracy?

---

## Shared infrastructure (build first — both methods need pieces of this)

### S1. Subset tags for the EXISTING mixed pool (do not resample)

The reweighting method needs subset tags on the *same 30k pairs* the mixed baseline trained on, or the comparison is confounded by a pool change.

**Do NOT** build the tagged pool by re-loading per-`data_dir` and re-shuffling — that produces a different row order than `load_hh_rlhf_mixed(seed=42)` and breaks apples-to-apples comparison.

**Instead, reconstruct tags by exact-match lookup:**
1. Load each of the four subsets separately: `load_dataset("Anthropic/hh-rlhf", data_dir=d, split="train")` for `d in [harmless-base, helpful-base, helpful-online, helpful-rejection-sampled]`.
2. Build a dict: `key = sha1(chosen + "\x00" + rejected)` → subset name. If a key appears in multiple subsets (duplicates), keep first match by the order above and count the collision.
3. Load the existing mixed pool exactly as the baseline did: `load_hh_rlhf_mixed(n=30000, split="train", seed=42)`.
4. Map each pair to its subset via the dict. Report: tag coverage (expect ~100%), collision count, unmatched count. If unmatched > 0.5%, STOP and report before proceeding.
5. Save tags to `results/mixed_pool_subset_tags_seed42.json` (row index → subset) so weights are reproducible without re-running the lookup.

### S2. Length definition

Use **tokenizer token count** of the response text (what the model actually sees), computed with the same DistilBERT tokenizer used in training. Define per pair: `dl_i = tok_len(chosen_i) - tok_len(rejected_i)`.
(Char-length correlations tracked token-length closely in Chunk 6's recomputation — token is the principled choice; report char-based as a robustness line in the final analysis only.)

### S3. Run matrix conventions (both methods)

- Train pool: the same mixed 30k (seed=42 sample) for every arm.
- Seeds: 42, 0, 1 (model init/training seeds), matching Chunk 9.
- Hyperparameters: identical to the mixed baseline runs (same LR, schedule, batch size, max_length=512 filter, early stopping on eval accuracy with the same patience).
- Eval: the fixed 1000-example (937 post-filter) seed=42 eval set. No changes.
- Output dirs: `results/reward_model_reweight_seed{42,0,1}/` and `results/reward_model_symloss_seed{42,0,1}/`. Never touch existing checkpoints.
- Report per checkpoint: eval accuracy, eval margin, pooled/chosen-only/rejected-only length-score r — same table format as Chunk 9 so tables concatenate.

---

## METHOD 1 — Subset-conditional length symmetrization (bias-aware reweighting)

### What it is, precisely

Per-example importance weights that make the length-gap distribution **symmetric around zero within every subset**. After weighting, "predict the winner from length" has zero expected advantage anywhere in the training distribution, so gradient descent cannot reduce loss via a pure length heuristic. Any length sensitivity that survives is content-mediated — which is exactly the decomposition claim: separating the removable (composition-driven) component from the residual.

This is deliberately **not** just "balance the four subsets." Equalizing subset proportions changes the mixture but leaves each subset's internal length-preference signal intact. Symmetrizing `dl` within subsets removes the signal at its source and handles all four subsets' different internal correlations in one scheme.

### Weight computation (closed-form, no tuning)

For the tagged 30k pool:
1. Within each subset `s`, take pairs with `dl != 0`. Compute quantile edges of `|dl|` → **K = 5** magnitude bins.
2. Each pair falls in cell `(s, k, sign(dl))`. Pairs with `dl == 0` get weight 1.0.
3. For each `(s, k)` cell pair: let `n_plus` = count with `dl > 0`, `n_minus` = count with `dl < 0`. Assign:
   `w_i = (n_plus + n_minus) / (2 * n_{sign(dl_i)})`
   (i.e., both signs get equal effective mass within every magnitude bin of every subset).
4. Cap: `w_i = min(w_i, 10.0)`. Renormalize all weights to mean 1.0 over the pool.
5. **Diagnostics to log (mandatory):** weighted fraction of chosen-longer overall and per subset (target: 50% ± 1%); weighted mean of `dl` (target ≈ 0); weight distribution (min/max/std, count at cap). If weighted chosen-longer is outside 48-52% overall, something is wrong — stop and report.
6. Save weights to `results/reweight_weights_seed42pool.json` keyed by row index.

**Robustness knob (run only after main results land, if time):** K=3 and K=10 variants at seed 42 only.

### Trainer modification (concrete)

TRL's `RewardTrainer` does not support per-example weights natively. Subclass:

```python
import torch.nn.functional as F
from trl import RewardTrainer

class WeightedRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        ).logits
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        ).logits
        diff = (rewards_chosen - rewards_rejected).squeeze(-1)
        per_pair_loss = -F.logsigmoid(diff)
        w = inputs["weight"].to(per_pair_loss.dtype)
        loss = (w * per_pair_loss).sum() / w.sum()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen,
                          "rewards_rejected": rewards_rejected}
        return loss
```

Plumbing requirements (these are the parts that silently break if skipped):
- Add the `weight` column to the dataset **before** tokenization mapping, and ensure the tokenize map function passes it through.
- Set `remove_unused_columns=False` in the config (or whitelist `weight`), otherwise HF strips it before the collator.
- Extend the reward data collator to stack `weight` into a float tensor alongside the tokenized fields. Verify with one debug batch: print `inputs.keys()` on the first training step and confirm `weight` is present with shape `(batch,)`.
- Eval examples: give them all `weight = 1.0` (the eval metric must remain unweighted for comparability).

### Match to TRL version installed

The exact field names (`input_ids_chosen`, etc.) and `compute_loss` signature must be checked against the installed trl 1.5.0's `reward_trainer.py` before writing code — read the source first (it's already been read once in this project for the column-mapping check; consult that). If the installed version's collator produces different key names, adapt to what's actually there rather than what's written above.

---

## METHOD 2 — Symmetric (sigmoid) loss on the Bradley-Terry margin

### Background, precisely

Bradley-Terry training is binary classification on the margin `z = r(chosen) − r(rejected)` with the logistic loss `ℓ(z) = −log σ(z)`. A loss is *symmetric* if `ℓ(z) + ℓ(−z) = const`; symmetric losses are provably robust to symmetric label noise (Ghosh et al. 2017; Charoenphakdee, Bao & Sugiyama 2019 — the latter is the citation that ties this to the Sugiyama weak-supervision line already in the proposal's reference base). Logistic loss is **not** symmetric: its unbounded tail means confidently-mispredicted pairs contribute unbounded gradient, which is precisely how a spurious-but-frequently-agreeing shortcut (length) keeps getting reinforced.

### The exact modification (one line)

```python
# baseline:  loss = -F.logsigmoid(diff).mean()
# symmetric: 
loss = torch.sigmoid(-diff).mean()
```

`ℓ(z) = σ(−z)` satisfies `ℓ(z) + ℓ(−z) = 1` exactly. Bounded in (0,1); gradient magnitude `σ(z)σ(−z) ≤ 1/4`, maximal at `z = 0` (healthy at init, where margins start near zero) and vanishing for large `|z|` in **both** directions — confidently-decided pairs stop contributing gradient instead of having their margins inflated further along whatever feature (including length) decides them.

### Honest mechanism note (put this in the paper too)

Symmetric loss targets **label-noise amplification**, not covariate shortcuts directly. The causal path to reduced length bias is indirect: bounded loss → no margin inflation on shortcut-consistent easy pairs → weaker incentive to grow a length-aligned score component. Prediction: Method 1 should dominate on bias reduction (it attacks the demonstrated mechanism — composition); Method 2 is the literature-anchored comparison. **If Method 2 alone substantially reduces the correlation, that's itself informative** — it says noise-amplification rather than composition was the dominant pathway, which would revise the Chunk 9 interpretation. Either outcome is a result.

### Training notes

- Same hyperparameters as baseline, but allow up to **2× max steps** with the same early-stopping rule (sigmoid loss can converge more slowly; do not tune LR — one controlled change at a time).
- Log the score-magnitude distribution (mean/std of raw rewards on eval): symmetric loss should visibly compress score magnitudes vs. baseline. This is a free sanity plot and a nice paper figure.
- No warm start needed (gradients are maximal at z≈0 where training begins).

---

## Evaluation plan and decision table (both methods)

Six runs minimum: {reweight, symloss} × seeds {42, 0, 1}. Optional 7th-9th: combined (weights + sigmoid loss) at all three seeds if wall-clock allows after the main six.

Comparison table (extend Chunk 9's format):

| arm | acc (937) | margin | r pooled | r chosen | r rejected |
with rows for harmless baseline (3 seeds), mixed baseline (3 seeds), reweight (3), symloss (3).

**Decision criteria (pre-registered — write results against these, don't move goalposts):**

| Outcome | Criteria |
|---|---|
| **Worked** | Mean \|pooled r\| ≤ 0.15 (≥ ~55% reduction from +0.32) AND mean accuracy ≥ 57.5% (within 2pts of mixed baseline) AND all three seeds agree in direction with r-range < 0.15 |
| **Bias-accuracy tradeoff** | r criterion met but accuracy 52-57.5% — still publishable as a quantified tradeoff |
| **Failed** | Mean \|pooled r\| > 0.25 (within noise of baseline) regardless of accuracy |
| **Inconclusive** | Seed r-range ≥ 0.15 (unstable) or mixed signs across seeds |

**Secondary checks (cheap, run after main table):**
- Re-run `score_framing_pairs.py` against the best-performing corrected checkpoint (code exists, minutes).
- If reweighting achieves its diagnostic targets (weighted chosen-longer ≈ 50%) but pooled r on eval stays high: report as "length signal not removable by composition rebalancing — residual is content-mediated or eval-distribution-driven." That is a substantive decomposition finding, not a failure.

---

## Implementation order and timeline

**Day 1:** Method 2 first — it's a one-line loss change with zero data plumbing. Implement, smoke-test one debug batch, launch symloss seed=42. While it trains: build S1 (tag reconstruction) and the weight computation + diagnostics.
**Day 2:** Verify weight diagnostics pass. Build `WeightedRewardTrainer` + collator plumbing, smoke-test one debug batch (confirm `weight` reaches `compute_loss`). Launch reweight seed=42.
**Days 2-5:** Sequence remaining runs (single MPS device — runs are serial; ~6 runs total at a few hours each).
**Day 6:** Comparison table, decision-table verdicts, secondary checks, write results summary to `results/phase3_decomposition_summary.md`.

Rationale for this order: Method 2 reaches first-result fastest (protects against wall-clock risk), while Method 1 — the more motivated method and the paper's primary correction — is built carefully during Method 2's training time rather than rushed.

**Total: ~1 week wall-clock.** Leaves 4-5 weeks for Phase 4 (light DPO comparison) + writing against the Aug 29 deadline.

---

## Failure-mode guards (read before coding)

1. **Weight column silently stripped** → the run trains as an unweighted baseline and looks like "reweighting did nothing." The debug-batch check on Day 2 exists to catch exactly this. Do not skip it.
2. **Tag mismatch on the mixed pool** → weights computed on wrong subset assignments. The S1 coverage/collision report exists to catch this. Do not proceed past unmatched > 0.5%.
3. **Comparing against a resampled pool** → confounds method effect with pool change. All arms train on the identical 30k rows.
4. **Moving decision thresholds after seeing results** → the table above is pre-registered; report against it verbatim, discuss nuances afterward.
5. **Version drift** → all work in `~/.venvs/rlhf-bias` (transformers 5.9.0, trl 1.5.0). Check trl's actual `reward_trainer.py` field names before subclassing.
