# Pre-registered predictions

Emi fills in every "Prediction" cell, then commits this file **before** E1's analysis runs.
The commit timestamp is the pre-registration record, so don't edit a prediction after its
experiment has run. Add a dated note below the table instead.

## Hypotheses

- **H0, prompt mix:** the pooled correlation reflects which prompts are in the eval set,
  not a preference between the two responses to the same prompt.
- **H1, length shortcut:** the reward model uses length itself, independent of content.
- **H2, refusal confound:** the reward model detects refusals; length correlates only
  because refusals are short.
- **H3, both:** a mix of H1 and H2, measured by how much the length effect shrinks once
  refusal is controlled.

## Thresholds

- An effect is **near zero** when its 95% bootstrap CI overlaps the range of the three
  untrained models on the same metric.
- An effect **persists** when its 95% CI excludes the untrained range on every training seed.
- (Emi: add or change thresholds here before committing.)

## Predictions

| Test | H1 predicts | H2 predicts | Prediction (Emi) | Confidence (Emi) |
| --- | --- | --- | --- | --- |
| E1: within-pair r, mixed models | Positive, outside untrained range | Positive, outside untrained range (H0: inside it) | | |
| E1: within-pair r, harmless-only models | Negative, outside untrained range | Negative, outside untrained range | | |
| E1: longer-wins rate, mixed models (humans: ~53%) | Above 53% | Above 53% | | |
| E1: within-pair r, reweight vs mixed | Similar | Similar | | |
| E1: which eval subset carries the effect | All subsets | Mostly harmless-base | | |
| E2: within-pair bias in both-comply pairs | Persists | Near zero | | |
| E2: within-pair bias in one-refuses pairs | Similar to both-comply | Carries most of the effect | | |
| E2: length coefficient once refusal is added | Barely shrinks | Shrinks toward zero | | |
| E2: refusal gap in the length-reweighted pool | Not relevant | Survives reweighting | | |
| E3: length bias across harmless share | Changes with share | Changes with share and tracks refusal-chosen rate | | |
| E4: refusal-balanced reweighting | Little change | Large drop, well beyond ~15% | | |

## Notes after results (dated, never edit the table above)
