# Revision run log

Append-only. Newest entry at the top. One entry per task, written by whoever did it.
Numbers here are copied from committed files; cite the file.

## Template

```
## YYYY-MM-DD: E<n> <short title>
- Who: Claude Code (Mac | RunPod) / planner / Emi
- Spec: revision/E<n>_SPEC.md @ <commit>
- Commit(s):
- What ran:
- Outputs:
- Headline numbers (from <committed file>):
- Flags / reproduction issues:
- Deviations / questions for Emi:
```

---

## 2026-10-23: Planning setup and repo audit

- Who: planner (claude.ai Grad school project), from a read-only clone
- Spec: none; this entry sets up `CLAUDE.md`, `revision/README.md`, `revision/PREDICTIONS.md`
  and `revision/E1_SPEC.md`
- Findings from reading the repo:
  1. All work since July is on `phase3-decomposition`; `main` is 24 commits behind it.
     `revision-sc4ai` branches from `phase3-decomposition` at b2796bd.
  2. Committed reweight results (`results/phase3_arms_length_corr.json`, RunPod rerun):
     r_pooled 0.266 / 0.283 / 0.317, mean +0.289; accuracy 0.5945 / 0.5945 / 0.5656, mean 0.585.
     Against the mixed baseline (`results/phase3_length_corr.json`): mean +0.339, accuracy 0.602,
     so about a 15% reduction with overlapping seed ranges. The +0.308 mean in the handoff doc is
     the earlier Colab run. Cite the committed numbers.
  3. Pool confound: `src/train_reward_model.py` draws the mixed pool with seed = training seed,
     but `experiments/decomposition/train_phase3.py` uses the seed-42 pool for every arm and seed.
     Only seed 42 is a matched comparison: mixed 0.349 vs reweight 0.266 (about a 24% drop).
     Two pool-matched baselines (seed-42 pool, training seeds 0 and 1, no weights) are added to
     the E3 batch.
  4. `load_hh_rlhf_eval_subset(by_subset=True)` loads subsets separately and may not reproduce the
     same 1,000 pairs, so E1 tags the eval set by hash lookup instead.
- Deviations / questions for Emi: whether to merge `phase3-decomposition` into `main` (fresh
  clones and repo syncs see `main` by default).
