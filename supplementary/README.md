# Supplementary Material

This directory contains the supplementary material accompanying the paper:

**Compositional Dual-Trigger Backdoors for Independent Prediction and Explanation Manipulation**

## Directory Structure

### `example_images/`

Some qualitative examples that complement the figures presented in the paper.

- `stageA/` — Explanation manipulation examples after Stage A training.
- `stageB/` — Examples illustrating the compositional backdoor after sequential training.

### `quantitative_results/`

Complete quantitative results for all experiments.

| File | Description |
|------|-------------|
| `accuracy_results.md` | Classification accuracies of the clean, Stage A, and Stage B models. |
| `stageA_results.md` | Stage A explanation preservation and explanation manipulation results. |
| `stageB_results.md` | Stage B prediction manipulation and joint-trigger evaluation results. |
| `retention_results.md` | Explanation backdoor retention after Stage B training. |

## Notes

The files in this directory contain the complete quantitative results for every evaluated model, dataset, and trigger composition.