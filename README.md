# FoR-Net

**FoR-Net: Learning to Focus on Hard Regions for Efficient Semantic Segmentation** :contentReference[oaicite:0]{index=0}

---

## Overview

FoR-Net is a lightweight semantic segmentation framework designed to focus on **hard and informative regions** such as thin structures and object boundaries.

Instead of relying on heavy global attention or computationally expensive architectures, FoR-Net introduces a simple yet effective **selector-driven Top-K mechanism** that explicitly emphasizes challenging spatial regions during training and inference.

The framework combines:

- Region-focused reasoning
- Top-K hard selection
- Multi-scale convolutional aggregation
- Lightweight decoding

to achieve efficient and practical semantic segmentation under limited computational resources.

---

## Features

- Selector-driven Top-K region activation
- Explicit hard-region reasoning
- Lightweight and efficient architecture
- Multi-scale receptive field aggregation
- Improved boundary and thin-structure segmentation
- Designed for resource-constrained environments

---

## Architecture

FoR-Net consists of:

1. **Backbone Encoder**
   - ResNet-101
   - Output stride = 8

2. **Selector Module**
   - Predicts an importance map
   - Identifies informative spatial regions

3. **Top-K Region Selection**
   - Explicit hard selection mechanism
   - Activates only the most important regions

4. **Multi-scale Reasoning**
   - Parallel convolution branches:
     - 1×1
     - 3×3
     - 5×5
     - 7×7 (dilated)

5. **Lightweight Decoder**
   - Refines and upsamples features for final prediction

---

## Experimental Setting

- Dataset: Cityscapes
- Backbone: ResNet-101
- Optimizer: AdamW
- Crop Size: 768×768
- Output Stride: 8
- Training Strategy:
  - Random scaling
  - Random cropping
  - Horizontal flipping

---

## Results

FoR-Net achieves competitive segmentation performance while maintaining a lightweight and efficient design.

| Method | Backbone | Val mIoU (%) |
|---|---|---|
| PSPNet | ResNet-101 | 78.4 |
| DANet | ResNet-101 | 78.8 |
| DeepLabV3+ | ResNet-101 | 79.3 |
| **FoR-Net (Ours)** | **ResNet-101** | **80.49** |

---

## Motivation

Recent segmentation models often depend on:

- heavy Transformer architectures
- expensive global modeling
- large GPU memory
- complex optimization strategies

FoR-Net explores a different direction:

> Explicitly focus computation on difficult and informative regions.

This work emphasizes practical usability and efficient reasoning under constrained hardware settings.

---

## Code

 **Code will be released soon.**

---

## Paper

Preprint available on arXiv.

---

## 🧪 Future Work

- Adaptive Top-K selection
- Dynamic region reasoning
- Integration with stronger backbones
- Evaluation on additional datasets

---

## Citation
  journal={arXiv preprint arXiv:2605.02764},
  year={2026}
}
