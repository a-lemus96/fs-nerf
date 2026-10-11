# Frequency Regularization for Few-Shot Neural Rendering

PyTorch implementation of frequency regularization techniques for improving Neural Radiance Fields (NeRF) in the few-shot setting.

## Overview

This repository explores two complementary regularization strategies for few-shot novel view synthesis using a SIREN-based NeRF (SiNeRF):

1. **Frequency Scheduler** — A frequency curriculum that constrains the angular frequency weights of sinusoidal activations, guiding optimization to explore low-frequency solutions early and progressively allow higher-frequency components. This provides an alternative to masking-based approaches over fixed positional encodings.

2. **Occlusion Regularization** — A geometry regularizer that penalizes undesirable density distributions along rays, encouraging the model to converge toward thin, surface-like geometry.

## Method

The combined training loss is:
```
L = L_reconstruction + α · L_frequency + β · L_occlusion
```

- `α` controls the frequency regularization weight (scheduled over training)
- `β` controls the occlusion regularization weight

## Setup

### Data

Download the LLFF scenes into `data/llff/` (~1.8 GB, git-ignored):
```bash
./scripts/download_llff.sh
```

### Weights & Biases

Create a `.env` file in the repository root (git-ignored) with your API key:
```
WANDB_API_KEY=<your-key>
```
Without it, run with `--debug` to disable wandb logging.

### Docker (recommended)

Requires the NVIDIA Container Toolkit.
```bash
docker image build -t fs-nerf . && docker image prune -f
docker compose up -d
docker compose exec fs-nerf bash
```

The container mounts the following:

| Host | Container | Notes |
|---|---|---|
| `./src` | `/workspace/src` | read-only, working directory |
| `./configs` | `/workspace/configs` | |
| `./data` | `/workspace/data` | read-only |
| `./output` | `/workspace/output` | results and wandb run files |
| `models` volume | `/models` | cached torch hub weights (e.g. VGG), kept across containers |

### Local
```bash
pip install -r requirements.txt
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html
```

## Training

Run from `src/`:
```bash
python train.py \
  --scene <llff_scene> \
  --n_imgs [3|6|9] \
  --n_iters <num_iterations> \
  --lr <learning_rate>
```

Key arguments:

| Argument | Description |
|---|---|
| `--model` | Model to be used for training (`nerf` or `sinerf`) |
| `--scene` | LLFF scene to be used for training |
| `--n_imgs` | Number of training views (`3`, `6`, or `9`) |
| `--n_iters` | Number of training iterations; falls back to `training.yaml` if not set |
| `--lr` | Initial learning rate for the optimizer; overrides the per-model default under `lr` in `training.yaml` (`nerf`, `sinerf`) if set |
| `--val_every` | Number of iterations between validation steps; falls back to `evaluation.yaml` if not set |
| `--out_dir` | Base directory for storing results (default `../output/`) |
| `--debug` | If set, run in debug mode (disables wandb logging) |
| `--render_only` | If set, load a pretrained model to render a video instead of training |

## Evaluation

Experiments are evaluated on the **LLFF dataset** using **PSNR** and **SSIM** metrics in the few-shot regime.

## Repository Structure
```
src/
├── core/
│   └── occlusion.py       # Occlusion regularizers (entropy, variance)
├── utils/
│   └── parser.py          # Argument parsing
...
```

## Thesis

This code accompanies a Master's thesis on frequency regularization for few-shot neural rendering. The thesis covers:
- Background on Neural Fields and NeRF
- Related work on spectral bias and frequency-based regularization
- Methodology for the frequency scheduler and occlusion regularization
- Experiments on LLFF with ablation studies

## Citation

*(To be added upon thesis publication)*
