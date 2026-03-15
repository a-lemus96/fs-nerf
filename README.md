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
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py \
  --data_dir <path_to_llff_scene> \
  --alpha <freq_reg_weight> \
  --beta <occlusion_reg_weight> \
  --reg [l1|l2]
```

Key arguments:

| Argument | Description |
|---|---|
| `--alpha` | Initial weight for frequency regularization |
| `--reg_ratio` | Fraction of iterations over which alpha is scheduled |
| `--reg` | Norm for frequency penalty (`l1` or `l2`) |
| `--beta` | Weight for occlusion regularization |
| `--depth_thres` | Depth threshold for sigma penalization |
| `--val` | Enable validation during training |
| `--val_rate` | Iterations between validation steps |

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
