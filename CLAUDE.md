# rlhf-bias-decomp: instructions for Claude Code

## What this project is now

A revision of *Flipping the Judge* for SC4AI'27 at AAAI-27 (deadline Nov 20, 2026 AoE).
The question: when a reward model trained on pooled HH-RLHF data looks length-biased,
what is it actually keying on, and why doesn't length reweighting remove it?

- Hub (status, the loop, file map): `revision/README.md`
- Current task: the newest `revision/E*_SPEC.md` marked "ready" in the hub's status table
- Run log: `revision/LOG.md`

## Roles

- **Emi** owns every decision, the predictions, and all paper prose. She approves each gate.
- **Planner** (a chat in Emi's claude.ai Grad school project) writes specs, interprets
  results, and keeps the plan doc current. It reads this repo by pulling the branch.
- **You (Claude Code)** implement and run specs exactly as written, then report.
  Don't interpret results beyond what a spec asks for.

## Rules

1. **Specs are contracts.** Don't change metrics, thresholds, seeds, the eval set, pools or
   hyperparameters. If a spec is wrong or impossible, stop, write the problem under
   "Deviations / questions" in `revision/LOG.md`, and ask Emi.
2. **Pre-registration.** Don't run an experiment's analysis until `revision/PREDICTIONS.md`
   holds Emi's predictions for it and is committed. Check `git log -- revision/PREDICTIONS.md`.
3. **Committed JSONs in `results/` are the only source of truth for numbers.** Never report
   a number that isn't in a committed file. Earlier runs produced numbers from unpersisted
   notebooks that later could not be reproduced; don't repeat that.
4. **Persist everything a run depends on:** training args (training_args.bin or an args
   JSON), a log, and a results JSON. The Phase 4 learning rate was never saved and a
   reviewer noticed.
5. **After every task,** add an entry at the top of `revision/LOG.md` (template inside),
   update the status table in `revision/README.md`, and commit them with the results.
   Commit messages start with the experiment id: `E1: ...`.
6. **Content safety.** No raw HH-RLHF text in results files, logs or commit messages.
   Refer to pairs by eval row index.

## Fixed definitions

- **Eval set:** `load_hh_rlhf_eval_subset(n=1000, split="test", seed=42)`, 937 pairs after
  TRL's >512-token filter. Never change it.
- **Scoring:** reuse `build_eval_records`, `score_texts`, `pearson` and `get_device` from
  `experiments/decomposition/eval_length_correlation.py`. Don't reimplement scoring.
- **Length:** response-only token count (text after the final `"\n\nAssistant:"`,
  `add_special_tokens=False`).
- **Seeds:** 42, 0, 1.
- **Base model:** `distilbert-base-uncased`, Bradley-Terry loss, lr 1e-5, batch 16,
  up to 3 epochs, early stopping on eval accuracy (patience 3), max length 512.

## Git

- Work on branch `revision-sc4ai` (branched from `phase3-decomposition` at b2796bd).
  Never push to `main`, never force-push, never rewrite history.
- `results/*` and `experiments/*` are gitignored by pattern, so new code and result files
  need `git add -f <path>`. Never add checkpoints (`*.safetensors`, `*.bin`, model dirs).
- Never delete or overwrite a checkpoint; new runs write to new directories.

## Environments

- **Mac (M3 Pro, 18 GB):** venv `~/.venvs/rlhf-bias`. fp32 only (bf16 triggers a ~40x
  Accelerate BLAS slowdown). MPS or CPU. Untrained baselines run on CPU so they are
  bit-reproducible.
- **RunPod:** network volume repo at `/workspace/persist/rlhf-bias-decomp`. Pinned:
  transformers 5.9.0, trl 1.5.0, datasets 4.8.5, accelerate 1.13.0, scipy 1.17.1,
  numpy 2.4.6. Template runner: `experiments/decomposition/run_phase3_arms_runpod.sh`.

## Checkpoints (confirm locations in `results/checkpoint_manifest.json`, built by E1 step 0)

| Arm | Directories | Training pool |
| --- | --- | --- |
| harmless-only | `results/reward_model_baseline`, `_baseline_seed0`, `_baseline_seed1` | unshuffled prefix, 100% harmless-base |
| mixed | `results/reward_model_mixed_seed{42,0,1}` | shuffled full corpus, **pool seed = training seed** |
| reweight, symloss, combined | `results/reward_model_{arm}_seed{42,0,1}` | **seed-42 mixed pool for every seed**; on the RunPod volume |

## Known pitfalls

- `load_hh_rlhf_subset` is an unshuffled prefix: 100% harmless-base for n ≤ 42,537.
  Never use it to build a new pool.
- `load_hh_rlhf_eval_subset(by_subset=True)` loads subsets separately and may shuffle into a
  different 1,000 pairs. Get subset tags by sha1 hash lookup, as in
  `experiments/decomposition/build_subset_tags.py`.
- Mixed baselines use a different pool per seed, but every Phase 3 arm uses the seed-42
  pool. Only seed 42 is a matched baseline-vs-arm comparison.
- The position-bias swap test is zero by construction for an independent scorer.
  Don't run it.
