# Neural Sinkhorn Gradient Flow (NSGF) and NSGF++

This repository provides a reference implementation of the **Neural Sinkhorn Gradient Flow** (NSGF) and its two‑phase extension **NSGF++** as described in the paper *Neural Sinkhorn Gradient Flow*.

## Directory Structure
```
├─ data/                # Datasets (MNIST, CIFAR‑10) – downloaded automatically via torchvision
├─ src/                 # Core library
│   ├─ __init__.py
│   ├─ utils.py         # Seed setting & TensorBoard logger
│   ├─ sinkhorn.py      # Wrapper around geomloss.SinkhornLoss
│   ├─ particles.py     # Euler integration utilities
│   ├─ models/          # Neural network definitions (MLP, UNet, TimePredictor)
│   ├─ training/        # NSFG and NSGF++ trainers
│   └─ inference/       # Generation scripts for NSGF and NSGF++
├─ experiments/         # End‑to‑end training/evaluation scripts
│   ├─ synthetic_2d.py # 2‑D synthetic benchmark
│   ├─ mnist.py         # MNIST generation & evaluation
│   ├─ cifar10.py       # CIFAR‑10 generation & evaluation
│   └─ metrics.py       # Wasserstein, FID, IS calculators
├─ configs/             # YAML configuration files for each experiment
├─ results/             # Generated figures & tables
├─ README.md            # **You are here**
└─ requirements.txt     # Pinned dependencies
```

## Installation
```bash
# Create a conda environment (optional but recommended)
conda create -n nsfg python=3.9 -y
conda activate nsfg

# Install dependencies (CUDA 11.7 compatible PyTorch)
pip install -r requirements.txt
```

## Quick Start
### 1. Synthetic 2‑D experiment
```bash
python experiments/synthetic_2d.py --config configs/nsfg_2d.yaml --device cpu
```
Trains an MLP velocity field on an 8‑gaussian mixture, generates samples and reports the 2‑Wasserstein distance.

### 2. MNIST generation (NSGF++)
```bash
python experiments/mnist.py --config configs/nsfgpp_mnist.yaml --device cuda
```
Trains the three components of NSGF++ (velocity, phase‑transition predictor, straight‑flow) and evaluates FID/IS.

### 3. CIFAR‑10 generation (NSGF++)
```bash
python experiments/cifar10.py --config configs/nsfgpp_cifar.yaml --device cuda
```
Same pipeline as MNIST but with a larger UNet.

## Configuration Files
All hyper‑parameters are stored in `configs/*.yaml`.  Feel free to edit them to change learning rates, number of iterations, Sinkhorn blur/scaling, etc.

## Evaluation
The `experiments/metrics.py` module provides:
- 2‑Wasserstein distance (SciPy OT)
- FID and Inception Score (via `torchmetrics`)

Metrics are automatically logged to TensorBoard and saved under `results/`.

## Reproducibility
- Random seeds are fixed via `src.utils.set_seed`.
- Deterministic CUDA operations are enabled.
- All checkpoints, logs and generated samples are stored in `results/` and `logs/`.

## License
This code is provided for educational purposes.  See the original paper for citation details.
