"""
Phase 3 reward-model trainer: symmetric (sigmoid) loss and/or per-example
importance weights, adapted to the ACTUAL trl 1.5.0 RewardTrainer API.

Version-specific facts this depends on (verified against the installed
trl/trainer/reward_trainer.py, 1.5.0):
  - The dataset is tokenized to `chosen_ids` / `rejected_ids` (NOT the
    `input_ids_chosen` names in older TRL / the spec draft).
  - `DataCollatorForPreference` stacks chosen then rejected into a SINGLE
    `input_ids` batch of 2*B rows; the model is called once and the logits
    are split with `torch.chunk(..., 2)`.
  - Signature columns are ["chosen_ids", "rejected_ids", "margin"], so a
    `weight` column is stripped unless `remove_unused_columns=False`.
  - eval_accuracy (used for early stopping / best-model selection) is produced
    inside `compute_loss` via `self._metrics`; we replicate that block exactly
    so behaviour is unchanged for the metric path.

Loss:
  - baseline logistic:  -logsigmoid(diff)
  - symmetric sigmoid:   sigmoid(-diff)          (l(z)+l(-z)=1, bounded)
Weights (optional): loss = sum(w * per_pair) / sum(w). If no `weight` column
is present (e.g. eval), it degrades to the unweighted mean -> identical to the
stock trainer, keeping the eval metric comparable.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.trainer.reward_trainer import DataCollatorForPreference, RewardTrainer


@dataclass
class DataCollatorForWeightedPreference(DataCollatorForPreference):
    """DataCollatorForPreference + a per-pair `weight` float tensor (B,)."""

    def torch_call(self, examples):
        output = super().torch_call(examples)
        if "weight" in examples[0]:
            output["weight"] = torch.tensor(
                [example["weight"] for example in examples], dtype=torch.float
            )
        return output


class Phase3RewardTrainer(RewardTrainer):
    """RewardTrainer with optional symmetric loss and optional per-pair weights.

    Pass `symmetric=True` for Method 2 (sigmoid loss). A `weight` column in the
    train dataset (Method 1) is consumed automatically. `remove_unused_columns`
    MUST be False in the config or the weight column is stripped before the
    collator (spec failure-mode guard #1).
    """

    def __init__(self, *args, symmetric: bool = False, **kwargs):
        self.symmetric = symmetric
        self._debug_printed = False
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        weight = inputs.pop("weight", None)

        # One-time debug-batch check (spec: confirm `weight` reaches compute_loss).
        if not self._debug_printed and mode == "train":
            keys = sorted(k for k in inputs if k != "use_cache")
            wshape = tuple(weight.shape) if weight is not None else None
            print(
                f"[debug-batch] compute_loss keys={keys} weight_shape={wshape} "
                f"symmetric={self.symmetric}",
                flush=True,
            )
            self._debug_printed = True

        inputs["use_cache"] = False
        outputs = model(**inputs)

        rewards_chosen, rewards_rejected = torch.chunk(outputs.logits.squeeze(-1), chunks=2)
        diff = rewards_chosen - rewards_rejected

        if self.symmetric:
            per_pair = torch.sigmoid(-diff)
        else:
            per_pair = -F.logsigmoid(diff)

        if weight is not None:
            w = weight.to(per_pair.dtype)
            loss = (w * per_pair).sum() / w.sum()
        else:
            loss = per_pair.mean()

        if self.args.center_rewards_coefficient is not None:
            loss = loss + self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected) ** 2
            )

        # --- metric logging: replicate stock RewardTrainer so eval_accuracy,
        # eval_margin, reward stats are produced for early stopping / logging ---
        if mode == "train":
            num_tokens_in_batch = (
                self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        with torch.no_grad():
            all_rewards = self.accelerator.gather(outputs.logits)
            self._metrics[mode]["min_reward"].append(all_rewards.min().item())
            self._metrics[mode]["mean_reward"].append(all_rewards.mean().item())
            self._metrics[mode]["max_reward"].append(all_rewards.max().item())

            mean_accuracy = (rewards_chosen > rewards_rejected).float().mean()
            mean_accuracy = self.accelerator.gather_for_metrics(mean_accuracy).mean().item()
            self._metrics[mode]["accuracy"].append(mean_accuracy)

            mean_margin = (rewards_chosen - rewards_rejected).mean()
            mean_margin = self.accelerator.gather_for_metrics(mean_margin).mean()
            self._metrics[mode]["margin"].append(mean_margin.item())

        return (loss, outputs) if return_outputs else loss
