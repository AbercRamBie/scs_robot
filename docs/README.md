# SCS Robot (Simple Overview - Early work)

Architecture blueprint: see ../ARCHITECTURE.md for a compartmentalized target layout and migration order.

Detailed ROS 2 migration steps: see ./ROS2_MIGRATION_GUIDE.md.

This project explores **semantic communication for robot navigation**.

In plain words: instead of sending a full map over a noisy channel, the model learns a compact representation that keeps only task-relevant information for navigation decisions.

## What the model predicts

For each generated occupancy grid, the model predicts one of 4 actions:

- `0`: forward
- `1`: left
- `2`: right
- `3`: stop

## How the pipeline works

1. A synthetic occupancy grid is generated.
2. An encoder compresses the grid into a small latent vector.
3. Noise is added through an AWGN channel (controlled by SNR).
4. A decoder predicts the navigation action.
5. Training uses a Variational Information Bottleneck (VIB) objective.

## Project structure

- `run_train.py`: main script to train and run SNR mismatch evaluation.
- `src/data/occupancy.py`: synthetic occupancy-grid dataset and dataloaders.
- `src/loss/vib.py`: reparameterization trick and VIB loss.
- `src/channel/awgn.py`: AWGN channel module.
- `src/train/trainer.py`: training loop with Weights & Biases logging.
- `src/evaluation/separation_baseline.py`: separation-based baseline experiment.
- `src/evaluation/results.py`: plotting script for experiment figures.
- `checkpoints/`: saved model weights.
- `results/`: JSON and plot outputs.
- `wandb/`: experiment logs.

## Quick setup

Use Python 3.10+ (3.11 also works in most cases).

Install dependencies:

```bash
pip install torch numpy matplotlib scipy tqdm wandb
```

(Optional) log in to Weights & Biases:

```bash
wandb login
```

## Run

### 1) Train semantic model + mismatch evaluation

```bash
python run_train.py
```

Expected artifacts:

- model checkpoint(s) in `checkpoints/`
- mismatch evaluation JSON in `results/mismatch_results.json`

### 2) Run separation baseline comparison

```bash
python src/evaluation/separation_baseline.py
```

Expected artifact:

- `results/baseline_comparison.json`

### 3) Generate plots

```bash
python src/evaluation/results.py
```

Expected artifacts:

- PNG/PDF figures in `results/`

## Current note about this repo snapshot

The training scripts import these files:

- `src/models/encoder.py`
- `src/models/decoder.py`
- `src/models/channel_layer.py`
- `src/train/config.py`

In the current workspace snapshot, these files are not present as Python source files.
So training will fail until they are restored.

If you have another branch or commit with those files, bring them back first.

## Outputs at a glance

- `results/mismatch_results.json`: semantic model accuracy vs test SNR.
- `results/baseline_comparison.json`: semantic vs separation vs random baseline.
- `wandb/`: detailed training metrics per run.

## One-line summary

This repo is a compact research prototype showing how semantic JSCC + VIB can improve task-level robot decision reliability under noisy channels.
