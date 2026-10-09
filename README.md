# FoR-Net: Focus-on-Regions Network for Semantic Segmentation

[**Paper (arXiv)**](https://arxiv.org/abs/2605.02764)

## Overview

FoR-Net (Focus-on-Regions Network) is a semantic segmentation framework designed to enhance informative spatial regions through a selector-driven Top-K mechanism and multi-scale contextual reasoning.

Built upon a ResNet-101 backbone, FoR-Net identifies important spatial locations using a learned importance map and performs region-focused feature enhancement with multiple receptive fields.

## Architecture

FoR-Net consists of the following components:

- **Region Selector:** Predicts spatial importance scores to identify informative regions.
- **Top-K Region Selection:** Applies branch-specific feature masking to prioritize informative spatial locations.
- **Multi-Scale Reasoning:** Employs parallel convolution branches with receptive fields of 1×1, 3×3, 5×5, and 7×7.
- **Global Context Module:** Incorporates global contextual information.
- **Segmentation Decoder:** Fuses high-level semantic features with low-level spatial information.

## Experimental Results

Performance on the Cityscapes validation set, as reported in the paper:

| Backbone | mIoU (%) | Parameters | GFLOPs |
|---|---:|---:|---:|
| ResNet-101 | 80.5 | 55.72M | 243.6 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from fornet import FoRNet

model = FoRNet(num_classes=19, pretrained=False)
model.eval()

x = torch.randn(1, 3, 128, 128)

with torch.no_grad():
    logits, sel_logit, masks = model(x)

print(logits.shape)
```

## Repository Structure

```text
FoR-Net/
├── fornet/
│   ├── __init__.py
│   ├── model.py
│   └── loss.py
├── README.md
└── requirements.txt
```

The repository includes the model architecture and loss functions. Training and evaluation scripts are not included.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{chan2026fornet,
  title={FoR-Net: Focus-on-Regions Network for Semantic Segmentation},
  author={Chan, Sheng-Wei and Pan, Hsin-Jui and Chiang, Jen-Shiun},
  journal={arXiv preprint arXiv:2605.02764},
  year={2026}
}
```
