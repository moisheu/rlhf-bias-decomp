# E1 spec: within-pair re-scoring of existing checkpoints

**Status:** ready. **No training.** Read `CLAUDE.md` first; its rules apply throughout.

## Why

The JuDGE reviewers objected that the pooled metric (Pearson r between score and response
length over all 1,874 responses) mixes two things: a preference between the two responses
to one prompt, and the fact that harmless prompts draw short responses. An untrained head
already reaches |r| = 0.22 on it.

E1 measures length bias **within pairs**, where the prompt is shared and cancels out. It
answers one question: is there a within-pair length preference at all (H1 or H2), or was
the pooled number measuring prompt mix (H0)?

## Gate

Steps 0 to 2 may run now. **Don't run step 3 (the full run) until `revision/PREDICTIONS.md`
has Emi's E1 predictions and is committed.** If it isn't, stop after step 2 and say so.

## Models

| Arm | Labels | Required | Committed numbers to reproduce |
| --- | --- | --- | --- |
| harmless | `harmless_seed{42,0,1}` | yes | `results/harmless_length_corr.json` |
| mixed | `mixed_seed{42,0,1}` | yes | `results/phase3_length_corr.json` |
| reweight | `reweight_seed{42,0,1}` | yes | `results/phase3_arms_length_corr.json` |
| untrained | `untrained_seed{42,0,1}` | yes | `results/untrained_length_correlation.json` |
| symloss, combined | `{arm}_seed{42,0,1}` | no; include with `--include-unstable` | `results/phase3_arms_length_corr.json` |

Checkpoint directories are listed in `CLAUDE.md`. Untrained models are not checkpoints:
build each with `torch.manual_seed(seed)` immediately followed by
`AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)`,
on CPU, exactly as `experiments/untrained_length_baseline.py` does.

## Step 0: checkpoint inventory

1. Find every required checkpoint directory. The harmless and mixed ones should be under
   `results/` on the Mac; the reweight ones are on the RunPod network volume.
2. If the reweight checkpoints are only on RunPod, copy the three directories to the Mac's
   `results/` (about 270 MB each) so every model is scored on the same hardware. If copying
   isn't possible, stop and ask Emi; don't score some models on a different machine
   without recording it.
3. Write `results/checkpoint_manifest.json`: for each label, the path, the machine, the
   sha256 of `model.safetensors` (or `pytorch_model.bin`), whether `training_args.bin`
   exists, and the seed recorded in it when readable.
4. If any required trained checkpoint can't be found, stop and report which.

## Step 1: code

**1a. One backward-compatible change** in
`experiments/decomposition/eval_length_correlation.py`: in `build_eval_records`, add
`"row_idx": i` to each record, where `i` is the row's index in `eval_ds`
(use `enumerate`). Change nothing else. Existing callers ignore the new key.

**1b. New script** `experiments/revision/e1_within_pair.py` (and an empty `__init__.py`),
run as `python -m experiments.revision.e1_within_pair [--include-unstable] [--only LABEL ...]`.
Import `build_eval_records`, `score_texts`, `pearson` and `get_device`; don't reimplement them.

**Eval-set subset tags.** Don't reload the eval set with `by_subset=True` (it may shuffle
into different pairs). Instead:
1. Load each of the four HH-RLHF subsets' **test** split separately
   (`load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="test")`).
2. Build `sha1(chosen + "\x00" + rejected) -> subset`, using the same key function and
   tie-break order as `experiments/decomposition/build_subset_tags.py`.
3. Map each of the 1,000 rows of `load_hh_rlhf_eval_subset(n=1000, split="test", seed=42)`.
   Coverage must be 100%; if any row is unmatched or collides across subsets, report it.
4. Save `results/eval_subset_tags_seed42.json`: row_idx -> subset, plus composition counts
   for the 1,000 rows and for the 937 kept pairs. For reference, earlier notes record the
   1,000-row composition as about 26.5% harmless-base, 27.4% helpful-base,
   12.7% helpful-online and 33.4% helpful-rejection-sampled.

**Scoring and reproduction check.** For each model, score chosen and rejected texts with
`score_texts`, then recompute `accuracy` and `r_pooled` exactly as
`eval_length_correlation.evaluate_checkpoint` does and compare them with the committed values.

| Absolute difference in accuracy or r_pooled | Status | Action |
| --- | --- | --- |
| ≤ 0.002 | ok | continue |
| > 0.002 and ≤ 0.01 | flag | continue; record it in the JSON and the log |
| > 0.01 | stop | skip this model's metrics; report it (likely the wrong checkpoint or environment drift) |

Untrained models on CPU should match their committed `pooled_r` to 1e-6. If they don't,
report it.

**Per-pair scores.** Write `results/e1_pair_scores.csv` with columns
`label, arm, seed, row_idx, subset, rlen_chosen, rlen_rejected, score_chosen, score_rejected`.
No text. E2 and E5 reuse this file instead of rescoring.

## Step 1c: metrics

Per model, on all 937 pairs and separately within each eval subset. Definitions:
`margin = score_chosen − score_rejected`; `dlen = rlen_chosen − rlen_rejected`.

| Metric | Definition |
| --- | --- |
| `accuracy` | mean(margin > 0) |
| `r_within` | Pearson r(margin, dlen) |
| `rho_within` | Spearman ρ(margin, dlen) (`scipy.stats.spearmanr`) |
| `longer_wins_rate` | over pairs with dlen ≠ 0 and margin ≠ 0: mean(sign(margin) == sign(dlen)), i.e. how often the model scores the longer response higher. Report n used and the counts excluded for dlen = 0 and for margin = 0 |
| `acc_longer_chosen` | accuracy on pairs with dlen > 0 (the human chose the longer response) |
| `acc_shorter_chosen` | accuracy on pairs with dlen < 0 |
| `length_gap` | acc_longer_chosen − acc_shorter_chosen |
| `r_pooled` | the old metric, kept for reference |

Once per eval set (a data property, identical for every model): `human_longer_chosen_rate`
= mean(dlen > 0) over dlen ≠ 0, overall and per subset, plus the count of dlen = 0 pairs.

**Uncertainty.** 1,000 bootstrap resamples of pairs, using
`np.random.default_rng(20261023)`. Draw the resample indices **once** and reuse them for
every model, so comparisons between models are paired. Report the percentile 95% CI for
accuracy, r_within, rho_within, longer_wins_rate and length_gap, overall and per subset.
Per subset, resample within that subset's pairs. Some subsets have only about 120 pairs;
report n beside every CI.

**Per-arm summary.** For each arm and metric (overall and per subset): the mean over the
three seeds, plus the minimum and maximum.

## Step 1d: outputs

1. `results/e1_within_pair.json`, with this structure:

```json
{
  "spec": "revision/E1_SPEC.md",
  "code_commit": "<git sha the run used>",
  "created": "<ISO date>",
  "device": "<cpu|mps|cuda per model>",
  "eval": {"n_pairs": 937, "n_dropped": 63, "subset_counts_937": {},
           "human_longer_chosen_rate": {"overall": 0.0, "by_subset": {}}, "n_dlen_zero": 0},
  "bootstrap": {"n": 1000, "rng_seed": 20261023},
  "models": [
    {"label": "mixed_seed42", "arm": "mixed", "seed": 42, "path": "", "checkpoint_sha256": "",
     "repro": {"accuracy": 0.0, "accuracy_committed": 0.0, "r_pooled": 0.0,
               "r_pooled_committed": 0.0, "status": "ok|flag|stop"},
     "overall": {"accuracy": {"value": 0.0, "ci": [0.0, 0.0]}, "r_within": {}, "rho_within": {},
                 "longer_wins_rate": {"value": 0.0, "ci": [0.0, 0.0], "n": 0},
                 "acc_longer_chosen": 0.0, "acc_shorter_chosen": 0.0, "length_gap": {}, "r_pooled": 0.0},
     "by_subset": {"harmless-base": {"n_pairs": 0}}}
  ],
  "arms": {"mixed": {"overall": {"r_within": {"mean": 0.0, "min": 0.0, "max": 0.0}}, "by_subset": {}}}
}
```

2. `results/e1_pair_scores.csv` and `results/eval_subset_tags_seed42.json` (above).
3. `revision/E1_RESULTS.md`, generated by the script from the JSON, never typed by hand:
   - the reproduction table;
   - the eval subset composition;
   - one per-arm table (rows: arms; columns: accuracy, r_within, rho_within,
     longer_wins_rate, length_gap and r_pooled, each as the seed mean with the [min, max]
     seed range);
   - the same table for each eval subset;
   - a per-model table with 95% CIs.
4. `results/e1_within_pair.png`: r_within per model with 95% CI whiskers, grouped by arm,
   with the untrained models' range shaded and a line at zero.

## Step 2: smoke test

Run `--only mixed_seed42 untrained_seed42`. Check:
- 937 pairs, 63 dropped;
- subset tags at 100% coverage;
- both models' reproduction status is ok;
- the CSV and JSON are written with the expected keys.

A sanity value from earlier notes: the old harmless_seed42 model's within-pair correlation
was about −0.47, measured with a different length unit. Expect strongly negative there,
but don't treat it as a pass/fail check.

## Step 3: full run (after the gate)

Run all required models. With `--include-unstable`, also run symloss and combined; their
numbers go in the appendix only.

## Step 4: commit and report

1. `git add -f` the new script, the `eval_length_correlation.py` change, the four result
   files, `revision/E1_RESULTS.md`, `results/checkpoint_manifest.json`, the `LOG.md` entry and
   the updated status row in `revision/README.md`.
2. Commit with a message starting `E1:`, then push `revision-sc4ai`.
3. In the log entry, include the reproduction table, the subset composition, the per-arm
   overall table, and every flag or deviation.

## Don't

- Interpret results beyond generating the tables, or edit `PREDICTIONS.md`.
- Change the eval set, drop pairs beyond the existing >512-token filter, or retrain anything.
- Put response text anywhere in the outputs.
