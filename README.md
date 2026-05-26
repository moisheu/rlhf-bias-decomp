# rlhf-bias-decomp

Research project investigating bias decomposition in Reinforcement Learning from Human Feedback (RLHF).

The goal is to understand and decompose the sources of bias present in human preference datasets used for RLHF training — including position bias, verbosity bias, and other annotator artifacts — and to measure how these biases propagate into reward models and fine-tuned language models.

## Datasets

- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) — helpfulness and harmlessness preference pairs
- [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) — Reddit-derived human preference data
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) — large-scale instruction-following feedback

## Setup

```bash
# Activate the virtual environment
source ~/.venvs/rlhf-bias/bin/activate

# Launch Jupyter
jupyter notebook
```

## Structure

```
data/           Raw and processed datasets
notebooks/      Exploratory analysis and experiments
src/            Reusable source code and utilities
experiments/    Experiment configs and run scripts
results/        Saved outputs, metrics, and figures
```
