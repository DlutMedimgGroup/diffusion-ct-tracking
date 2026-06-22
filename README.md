# Diffusion-Guided Anatomical Position Encoding for Dense Longitudinal CT Correspondence (Anonymous Submission)

This repository provides the anonymized implementation of our MICCAI submission:

**"Diffusion-Guided Anatomical Position Encoding for Dense Longitudinal CT Correspondence"**

---

## Overview

We propose a **diffusion-guided anatomical position encoding** framework for dense CT tracking **without explicit spatial registration or correspondence annotations**.

The framework consists of two stages:

- **Stage 1:** Diffusion-guided anatomical position embedding learning  
- **Stage 2:** Coarse-to-fine refinement for spatially precise correspondence  

The method is trained in an **unsupervised** manner using multi-subject CT data and evaluated on:

- DIR-Lab 4DCT (lung landmark tracking)  
- Deep Longitudinal Study (lesion tracking)

---

# 🚀 Quick Start (Run the Demo)

The fastest way to verify the code is:

1. Download demo data and pretrained weights  
2. Run Stage 1 inference  
3. Run Stage 2 inference  

The whole demo runs in minutes.

---

## Installation

### Requirements

- Linux  
- Python 3.12  
- CUDA 13.x  
- NVIDIA GPU (required)

Tested with:

- PyTorch 2.10 (CUDA 13.1)  
- MONAI 1.5.2  

---

### Install PyTorch

```bash
pip install torch torchvision torchaudio
```

### Install project dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Pretrained Weights and Demo Data

Pretrained checkpoints and a small demo dataset are provided via **GitHub Releases**.

---

## Step 1. Create a workspace

```bash
mkdir -p demo
cd demo
```

After this step, the current directory is treated as:

```
<PROJECT_ROOT>
```

---

## Step 2. Prepare folder structure

```bash
mkdir -p trained_models/demo_coarse
mkdir -p trained_models/demo_local
mkdir -p demo_data
```

Expected structure:

```
<PROJECT_ROOT>/
  trained_models/
    demo_coarse/
    demo_local/
  demo_data/
```

---

## Step 3. Download checkpoints and demo data

### Download checkpoints and demo data

Run the helper script:

```bash
bash scripts/download_demo.sh
```

Move checkpoints:

```bash
mv stage1_coarse.pth trained_models/demo_coarse/
mv stage2_refine.pth trained_models/demo_local/
```

Unzip demo data:

```bash
unzip demo_data.zip -d demo_data
```

---

## Step 4. Edit dataset split file

Template:

```
configs/demo_dataset.json
```

⚠️ **You MUST update the paths** to match your local environment.

---

## Step 5. Run Stage 1

```bash
python run_inference.py --config configs/demo_coarse.toml
```

---

## Step 6. Run Stage 2

```bash
python run_inference.py --config configs/demo_local.toml
```

Results will be saved under:

    <PROJECT_ROOT>/inference_results/


---

# 🧠 Training

## Train Stage 1

Single GPU:

```bash
python run_training.py --config <your_stage1_config.toml>
```

Multi-GPU:

```bash
torchrun --nproc_per_node=<NUM_GPUS> run_training_ddp.py \
  --config <your_stage1_config.toml>
```

---

## Train Stage 2

Single GPU:

```bash
python run_training.py --config <your_stage2_config.toml>
```

Multi-GPU:

```bash
torchrun --nproc_per_node=<NUM_GPUS> run_training_ddp.py \
  --config <your_stage2_config.toml>
```

---

# 📌 Notes

- This repository is anonymized for double-blind review  
- Demo data is only for functional verification  
- Multi-GPU training uses PyTorch DDP  

---

# 📄 Citation

If you find this work useful, please consider citing our MICCAI submission.
