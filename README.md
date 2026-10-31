# SwiftMask-CES

## Cost-Efficient Segmentation under a Compute Budget

A lightweight foreground segmentation system developed for
CSL7360: Computer Vision Assignment 2.

## Approach

Track B: Fully Convolutional Network implemented from scratch
using PyTorch.

The system uses:
- Lightweight encoder-decoder architecture
- Depthwise separable convolutions
- Skip connections
- Pixel-wise segmentation loss
- CPU-optimized inference

## Objectives

- Foreground/background binary segmentation
- Average CPU inference time <= 50 ms
- Small model parameter file
- High mIoU and Boundary F1
- Optimized Cost-Efficient Segmentation score

## Repository Structure

- `src/ces/`: Main implementation
- `scripts/`: Training and evaluation scripts
- `models/`: Model checkpoints and final parameters
- `captured/`: Independently captured photographs
- `outputs/`: Generated predictions and visualizations
- `reports/`: Experimental results and final report
- `tests/`: Unit tests

## Dataset

DUTS-TR is divided into training and validation sets.
DUTS-TE is reserved exclusively for final evaluation.

Captured college/hostel photographs are used only for
independent testing.

## Execution

Training:

```bash
python scripts/train.py