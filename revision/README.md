# Revision for SC4AI'27

This folder is the execution side of the revision. Decisions, rationale and the schedule
live in the plan doc: https://claude.ai/artifact/7RNrg26zkSsi1N4JybptAp (private to Emi).

## The loop

1. The planner writes `E<n>_SPEC.md`; Emi reviews it and commits it here.
2. Emi tells Claude Code: "Read CLAUDE.md and revision/E<n>_SPEC.md, then do it."
3. Claude Code implements and runs it, then commits code, results, a `LOG.md` entry and an
   updated status table, and pushes `revision-sc4ai`.
4. Emi tells the planner chat "E<n> is pushed." The planner pulls, reads the results JSON and
   the log entry, checks them against `PREDICTIONS.md`, updates the plan doc, and writes the
   next spec.
5. Emi makes each gate call in the planner chat: Nov 2 story decision, Nov 4 E6 cutoff,
   Nov 8 experiment freeze.

## Status

| Exp | Spec | Status | Results | Commit |
| --- | --- | --- | --- | --- |
| E1 Within-pair re-scoring | `E1_SPEC.md` | spec ready; waiting on predictions | | |
| E2 Refusal decomposition | written after E1 lands | not started | | |
| E3 Mixture sweep + 2 pool-matched baselines | written with E2 | not started | | |
| E4 Refusal-balanced reweighting | only if E2 points to refusal | not started | | |
| E5 Aggregation check | after E3 | not started | | |
| E6 Larger backbone | only if started by Nov 4 | not started | | |

## Files

| File | What it is | Who writes it |
| --- | --- | --- |
| `PREDICTIONS.md` | Pre-registered hypotheses, predictions and thresholds | Emi, before E1 runs |
| `E<n>_SPEC.md` | The contract for one experiment | Planner |
| `E<n>_RESULTS.md` | Auto-generated tables from that experiment's JSON | Claude Code's script |
| `LOG.md` | Append-only record of every task, newest first | Whoever did the task |
