# Open-Vocabulary 3D Scene Understanding

<div align="center">

**Comparative Analysis of Open-Vocabulary 3D Scene Understanding using Enhanced SAM Masks and Multiple Segmentation Methods**

*M.Tech in Artificial Intelligence — Indian Institute of Science (IISc), Bengaluru*

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Works Summary](#works-summary)
- [Datasets and Evaluation](#datasets-and-evaluation)
- [Prerequisites](#prerequisites)
- [Work 1 — CAHMU: Context-Aware Hierarchical Mask Unifier](#work-1--cahmu-context-aware-hierarchical-mask-unifier)
- [Work 2 — Improved Gaussian Grouping](#work-2--improved-gaussian-grouping)
- [Work 3 — Segmentation over Sparse Voxel Rasterization (SVRaster)](#work-3--segmentation-over-sparse-voxel-rasterization-svraster)
- [Work 4 — Enhanced OpenGaussian](#work-4--enhanced-opengaussian)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Citation](#citation)

---

## Overview

This repository contains the full experimental codebase for the M.Tech. thesis project on **Open-Vocabulary 3D Scene Understanding**. The project systematically investigates how enhanced segmentation mask supervision and novel 3D learning objectives can lift the state-of-the-art on open-vocabulary, text-query-driven 3D instance segmentation over radiance-field representations.

The thesis introduces four interconnected contributions:

| # | Work | Description |
|---|------|-------------|
| **1** | **CAHMU** | A training-free, CLIP-driven hierarchical mask unifier that resolves multi-level SAM granularity conflicts |
| **2** | **Improved Gaussian Grouping** | Augments 3DGS identity encoding with contrastive, hypersphere, and Gaussian Semantic Tracing (GST) losses |
| **3** | **Segmentation over SVRaster** | First end-to-end object-feature learning over Sparse Voxel Rasterization, with contrastive, hypersphere, and Voxel Semantic Tracing (VST) losses |
| **4** | **Enhanced OpenGaussian** | Systematic mask-type and scene-cropping ablation of OpenGaussian with CAHMU-refined CLIP supervision |

---

## Repository Structure

```
Open-Vocabulary-3D-Scene-Understanding/
│
├── modified-gaussian-grouping/          # Work 2: Modified Gaussian Grouping codebase
│   ├── submodules/
│   │   ├── diff-gaussian-rasterization/
│   │   ├── simple-knn/
│   │   ├── fused-ssim/
│   │   ├── GroundingDINO/
│   │   ├── deva/
│   │   ├── sam-hq/
│   │   └── depth-anything-v2/
│   ├── script/
│   │   ├── download_models.sh
│   │   ├── prepare_pseudo_label.sh
│   │   └── train_render_eval.sh
│   ├── config/
│   │   └── GroundingDINO_SwinB_cfg.py   ← Download manually (see Work 2 setup)
│   ├── checkpoints/                     ← Populated by download_models.sh
│   │   ├── depth_anything_v2_vitg.pth
│   │   ├── depth_anything_v2_vitl.pth
│   │   ├── DEVA-propagation.pth
│   │   ├── groundingdino_swinb_cogcoor.pth
│   │   ├── sam_hq_vit_h.pth
│   │   └── sam_vit_h_4b8939.pth
│   ├── data/
│   │   ├── ramen/                       ← Scene images + COLMAP
│   │   ├── figurines/
│   │   ├── teatime/
│   │   └── label/
│   │       ├── ramen/                   ← JSON annotation files
│   │       ├── figurines/
│   │       └── teatime/
│   ├── json_to_masks.py
│   ├── environment.yml
│   └── runner_of_gaussian-grouping.sh
│
├── modified-svraster/                   # Work 3: modified-svraster codebase
│   ├── cuda/
│   ├── fused-ssim/
│   ├── GroundingDINO/
│   ├── sam-hq/
│   ├── scripts/
│   │   └── train_render_eval.sh
│   ├── cfg/
│   │   └── GroundingDINO_SwinB_cfg.py   ← Copy from Work 2 config/ (see Work 3 setup)
│   ├── checkpoints/                     ← Copy from Work 2 checkpoints/ (see Work 3 setup)
│   │   ├── groundingdino_swinb_cogcoor.pth
│   │   ├── sam_hq_vit_h.pth
│   │   └── sam_vit_h_4b8939.pth
│   ├── data/                            ← Copy entire data/ from Work 2 (see Work 3 setup)
│   │   ├── ramen/
│   │   ├── figurines/
│   │   └── teatime/
│   ├── environment.yml
│   └── runner_of_svraster.sh
│
├── modified-OpenGaussian/               # Work 1 (masks) + Work 4: modified-OpenGaussian
│   ├── submodules/
│   │   ├── ashawkey-diff-gaussian-rasterization/
│   │   ├── sam-langsplat/
│   │   └── sam-hq/
│   ├── assets/
│   │   └── text_features.zip
│   ├── scripts/
│   │   └── train_render_eval.sh
│   ├── ckpts/                           ← Copy from Work 2/3 checkpoints/ (see Work 4 setup)
│   │   ├── sam_hq_vit_h.pth
│   │   └── sam_vit_h_4b8939.pth
│   ├── data/
│   │   └── lerf_ovs/                   ← Scene images + COLMAP for Work 4
│   │       ├── ramen/
│   │       ├── figurines/
│   │       └── teatime/
│   ├── preprocess_sam_l.py
│   ├── preprocess_sam_hq.py
│   ├── preprocess_sam_u.py
│   ├── masks_visualizer.py
│   ├── train_normal.py
│   ├── crop_scene.py
│   ├── crop_images.py
│   ├── render_lerf_by_text.py
│   ├── environment.yml
│   └── runner_of_OpenGaussian.sh
│
└── README.md
```

> ⚠️ **Important — Data folder layout:**
> - Works 2 and 3 expect scene data directly under `data/<scene>/` (e.g. `data/ramen/`), with JSON label files under `data/label/<scene>/`.
> - Work 4 expects scene data under `data/lerf_ovs/<scene>/` (e.g. `data/lerf_ovs/ramen/`), with JSON label files under `data/lerf_ovs/label/<scene>/`.
> The codes in each of the works are written according to these paths; do **not** deviate from these structures.

---

## Works Summary

<div align="center">

```mermaid
flowchart TD
    S(["Input Scenes — ramen / figurines / teatime"])
    W1["Work 1 — CAHMUTraining-free mask unifier→ Unified single-level SAM masksvia OpenGaussian preprocessing scripts"]
    UM(["Unified Masks"])
    W2["Work 2 —Improved Gaussian Grouping"]
    W3["Work 3 —Introducing SVRaster Segmentation"]
    W4["Work 4 —Enhanced OpenGaussian"]

    S --> W1
    W1 --> UM
    UM --> W2 & W3 & W4
```

</div>

---

## Datasets and Evaluation

All experiments are conducted exclusively on the **LeRF-OVS** dataset, comprising three real-world tabletop scenes captured with the Polycam application: `figurines`, `teatime`, and `ramen`.

### Dataset Download

We have expanded upon the original LeRF-OVS collection and also provide the corresponding COLMAP data sourced from the [LangSplat](https://github.com/minghanqin/LangSplat) repository. All scenes used in this thesis — `figurines`, `ramen`, and `teatime` — are included.

> 📦 **[Download Expanded LERF Dataset and COLMAP Data](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing)**

After downloading, extract the archive and place the scenes in the appropriate directories for each work:

```
# For Works 2 and 3
modified-gaussian-grouping/data/ramen/
modified-gaussian-grouping/data/figurines/
modified-gaussian-grouping/data/teatime/

# For Work 4
modified-OpenGaussian/data/lerf_ovs/ramen/
modified-OpenGaussian/data/lerf_ovs/figurines/
modified-OpenGaussian/data/lerf_ovs/teatime/
```

### Scene Overview

| Scene | Object Density | Notable Objects |
|-------|---------------|-----------------|
| `ramen` | Medium | nori, sake cup, kamaboko, corn, spoon, egg, chopsticks, wavy noodles, bowl, napkin, etc. |
| `figurines` | High | jake, pikachu, rubber duck, pirate hat, waldo, tesla door handle, porcelain hand, etc. |
| `teatime` | Low–Medium | sheep, stuffed bear, coffee mug, cookies, apple, yellow pouf, dall-e brand, etc. |

### Fixed Evaluation Frames and Queries

Evaluation frames and text queries are fixed per scene across **all experiments** to ensure strictly comparable conditions.

- **ramen** queries: `nori`, `sake cup`, `kamaboko`, `corn`, `spoon`, `egg`, `onion segments`, `plate`, `napkin`, `bowl`, `glass of water`, `chopsticks`, `wavy noodles`
  — frames: 00006, 00024, 00060, 00065, 00081, 00119, 00128
- **figurines** queries: `jake`, `pirate hat`, `pikachu`, `rubber duck with hat`, `porcelain hand`, `red apple`, `tesla door handle`, `waldo`, `bag`, `toy cat statue`, `miffy`, `green apple`, `pumpkin`, `rubics cube`, `old camera`, `rubber duck with buoy`, `red toy chair`, `pink ice cream`, `spatula`, `green toy chair`, `toy elephant`
  — frames: 00041, 00105, 00152, 00195
- **teatime** queries: `sheep`, `yellow pouf`, `stuffed bear`, `coffee mug`, `tea in a glass`, `apple`, `coffee`, `hooves`, `bear nose`, `dall-e brand`, `plate`, `paper napkin`, `three cookies`, `bag of cookies`
  — frames: 00002, 00025, 00043, 00107, 00129, 00140

### Hardware Requirements

| Work | Scene | GPU |
|------|-------|-----|
| Work 2 | figurines, ramen | NVIDIA A6000 48 GB |
| Work 2 | teatime | NVIDIA A100 80 GB |
| Work 3 | all scenes | NVIDIA A100 80 GB |
| Work 4 | all scenes | NVIDIA A6000 48 GB |

---

## Prerequisites

All three codebases share the following common requirements:

- **CUDA 12.1** (`nvidia/label/cuda-12.1.0`)
- **Conda** (Miniconda or Anaconda)
- **GCC / G++** available at `/usr/bin/gcc` and `/usr/bin/g++`
- **GPU architecture flags:** `8.0` (A100) and/or `8.6` (A6000 / RTX 3090)

> ⚠️ Each work uses its own isolated conda environment. Do **not** share environments across works.

---

## Work 1 — CAHMU: Context-Aware Hierarchical Mask Unifier

**CAHMU** is a **training-free** preprocessing module. It resolves the granularity conflicts in multi-level SAM outputs using CLIP-driven objectness scoring, producing a clean, instance-discriminative single-level mask set. These unified masks serve as supervision for Works 2 and 4.

Work 1 mask generation is embedded inside the **OpenGaussian runner** (`runner_of_OpenGaussian.sh`). The relevant script is `preprocess_sam_u.py`, which runs the full CAHMU-unified SAM pipeline.

### Step-by-step

Navigate into `modified-OpenGaussian/` and run the following (also covered by `runner_of_OpenGaussian.sh`):

```bash
cd modified-OpenGaussian

# 1. Create and activate the environment
conda env create -f environment.yml
conda activate open_gaussian

# 2. Install CUDA toolkit and build tools
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# 3. Install submodules
pip install --no-build-isolation submodules/ashawkey-diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation submodules/sam-langsplat
pip install --no-build-isolation submodules/sam-hq

# 4. Unzip text features
cd assets && unzip text_features.zip && cd ..

# 5. Set up ckpts/ folder — copy SAM checkpoints from Work 2 or Work 3 (see Work 4 setup)
#    ckpts/sam_hq_vit_h.pth
#    ckpts/sam_vit_h_4b8939.pth

# 6. Generate CAHMU-Unified SAM masks (Work 1 output) for each scene
python preprocess_sam_u.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_u.py --dataset_path data/lerf_ovs/ramen --crop

python preprocess_sam_u.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_u.py --dataset_path data/lerf_ovs/figurines --crop

python preprocess_sam_u.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_u.py --dataset_path data/lerf_ovs/teatime --crop

# (Optional) Visualise the generated masks
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_u
```

> The `--crop` flag generates masks from black-background cropped renders (used in crop-setting ablations for Work 4).

**CAHMU key thresholds** (held fixed across all experiments):

| Parameter | Value |
|-----------|-------|
| CLIP temperature τ | 25.0 |
| Containment ratio | > 0.8 |
| Complexity–objectness band (κ, o) | (0.45, 0.85) |
| Pairwise histogram-correlation gate | (0.25, 0.75) |
| Dynamic split threshold θ_split | [0.4, 0.85] |
| Foreground-canvas inclusion | o > 0.1 |
| Vacuum-overlap ratio | r < 0.5 |
| Min residual mask area | 50 px |

---

## Work 2 — Improved Gaussian Grouping

Work 2 augments the original [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) framework with three novel loss terms applied over a 3DGS backbone:

- **Cont** — 2D + 3D contrastive loss
- **Hyp** — 2D + 3D hypersphere normalisation
- **GST** — KL distillation via Gaussian Semantic Tracing

**Best configuration: Exp 8** — CAHMU-unified masks on original images + Hyp loss → **41.72% mean mIoU**

### Installation

```bash
cd modified-gaussian-grouping

conda env create -f environment.yml
conda activate my_gaussian_grouping

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/fused-ssim
pip install --no-build-isolation ./submodules/GroundingDINO
pip install --no-build-isolation ./submodules/deva
pip install --no-build-isolation ./submodules/sam-hq
pip install --no-build-isolation ./submodules/depth-anything-v2

# Download segmentation model checkpoints into checkpoints/
bash script/download_models.sh
```

### GroundingDINO Config File

> ⚠️ The `wget` approach for the GroundingDINO config does not download the raw file correctly from GitHub. **Download the file manually** from the link below and place it in the `config/` folder:

1. Open: [https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py)
2. Click **Raw** and save the file as `GroundingDINO_SwinB_cfg.py`
3. Place it at:

```
modified-gaussian-grouping/config/GroundingDINO_SwinB_cfg.py
```

### Step 1 — Convert JSON annotations to masks

```bash
python json_to_masks.py --data_dir data/label/ramen
python json_to_masks.py --data_dir data/label/figurines
python json_to_masks.py --data_dir data/label/teatime
```

### Step 2 — Prepare pseudo-labels (mask supervision)

Six mask variants are prepared per scene by crossing three mask types × two image sources:

```bash
# ramen
bash script/prepare_pseudo_label.sh ramen 1 sam
bash script/prepare_pseudo_label.sh ramen 1 sam_hq
bash script/prepare_pseudo_label.sh ramen 1 sam_unified
bash script/prepare_pseudo_label.sh ramen crop sam
bash script/prepare_pseudo_label.sh ramen crop sam_hq
bash script/prepare_pseudo_label.sh ramen crop sam_unified

# figurines
bash script/prepare_pseudo_label.sh figurines 1 sam
bash script/prepare_pseudo_label.sh figurines 1 sam_hq
bash script/prepare_pseudo_label.sh figurines 1 sam_unified
bash script/prepare_pseudo_label.sh figurines crop sam
bash script/prepare_pseudo_label.sh figurines crop sam_hq
bash script/prepare_pseudo_label.sh figurines crop sam_unified

# teatime
bash script/prepare_pseudo_label.sh teatime 1 sam
bash script/prepare_pseudo_label.sh teatime 1 sam_hq
bash script/prepare_pseudo_label.sh teatime 1 sam_unified
bash script/prepare_pseudo_label.sh teatime crop sam
bash script/prepare_pseudo_label.sh teatime crop sam_hq
bash script/prepare_pseudo_label.sh teatime crop sam_unified
```

> **Note:** The preprocessing above generates all required mask folders (e.g. `object_mask_sam/`, `crop_object_mask_sam_hq/`, etc.) inside each scene directory under `data/`. These will be reused by Work 3 — see the [Work 3 data setup](#work-3-data-and-checkpoint-setup) section below.

### Step 3 — Train, render, and evaluate

**Mask-quality ablation** (Exps 1–6 per scene, no novel loss terms):

```bash
# ramen
bash script/train_render_eval.sh ramen output_sam              object_mask_sam               normal
bash script/train_render_eval.sh ramen output_crop_sam         crop_object_mask_sam          normal
bash script/train_render_eval.sh ramen output_sam_unified      object_mask_sam_unified       normal
bash script/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified  normal
bash script/train_render_eval.sh ramen output_sam_hq           object_mask_sam_hq            normal
bash script/train_render_eval.sh ramen output_crop_sam_hq      crop_object_mask_sam_hq       normal
```

**Loss-function ablation — Unified-Original baseline** (Exps 7–10):

```bash
bash script/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh ramen output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all
```

**Loss-function ablation — HQ-SAM Cropped baseline** (Exps 11–14):

```bash
bash script/train_render_eval.sh ramen output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh ramen output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh ramen output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh ramen output_crop_sam_hq_all crop_object_mask_sam_hq all
```

> Repeat all `train_render_eval.sh` calls above for `figurines` and `teatime` by substituting the scene name. See `runner_of_gaussian-grouping.sh` for the complete command listing.

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Iterations | 30,000 |
| Optimizer | Adam (default 3DGS schedule) |
| Identity feature classes | 256 |
| λ_GST | 50 |
| λ²ᵈ_cont | 5×10⁻⁴ |
| λ³ᵈ_cont | 2×10⁻² |
| λ²ᵈ_hyp | 10⁻³ |
| λ³ᵈ_hyp | 10⁻³ |
| GST update interval | 100 iters |
| GST views per update | 20 |

---

## Work 3 — Segmentation over Sparse Voxel Rasterization (SVRaster)

Work 3 introduces the **first end-to-end object-feature learning pipeline over** [SVRaster](https://github.com/theialab/svraster), requiring new per-voxel object features and semantic-tracing weights to be added directly to the CUDA rasteriser. The novel **Voxel Semantic Tracing (VST)** loss provides soft probabilistic KL distillation of multi-view semantic consistency.

**Best configuration: Exp 13** — HQ-SAM on cropped renders + VST loss → **45.53% mean mIoU**

### Installation

```bash
cd modified-svraster

conda env create -f environment.yml
conda activate svraster

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

pip install --no-build-isolation ./cuda/
pip install --no-build-isolation ./fused-ssim/
pip install --no-build-isolation ./GroundingDINO/
pip install --no-build-isolation ./sam-hq/
```

### GroundingDINO Config File

> ⚠️ Same as Work 2 — **download the raw config file manually** from GitHub and place it in the `cfg/` folder. The simplest approach is to copy it directly from Work 2:

```bash
# From the repository root
cp modified-gaussian-grouping/config/GroundingDINO_SwinB_cfg.py \
   modified-svraster/cfg/GroundingDINO_SwinB_cfg.py
```

Alternatively, download it manually from:
[https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py)

### Work 3 Data and Checkpoint Setup

The data preprocessing step for Work 3 is **identical** to Work 2. Once you have completed the preprocessing steps in Work 2, **copy the entire `data/` folder** directly into Work 3 instead of re-running preprocessing:

```bash
# From the repository root — copy Work 2 data (with all prepared masks) to Work 3
cp -r modified-gaussian-grouping/data modified-svraster/data
```

Work 3 also requires a `checkpoints/` folder containing three model weight files. Copy them from Work 2's `checkpoints/` folder (populated by `download_models.sh`):

```bash
# From the repository root
mkdir -p modified-svraster/checkpoints

cp modified-gaussian-grouping/checkpoints/groundingdino_swinb_cogcoor.pth \
   modified-svraster/checkpoints/

cp modified-gaussian-grouping/checkpoints/sam_hq_vit_h.pth \
   modified-svraster/checkpoints/

cp modified-gaussian-grouping/checkpoints/sam_vit_h_4b8939.pth \
   modified-svraster/checkpoints/
```

After these steps, the `modified-svraster/` directory should have:

```
modified-svraster/
├── checkpoints/
│   ├── groundingdino_swinb_cogcoor.pth
│   ├── sam_hq_vit_h.pth
│   └── sam_vit_h_4b8939.pth
├── cfg/
│   └── GroundingDINO_SwinB_cfg.py
└── data/
    ├── ramen/         (with all prepared mask folders)
    ├── figurines/
    └── teatime/
```

### Train, render, and evaluate

The `train_render_eval.sh` script for SVRaster takes an additional per-scene **bound scale** argument controlling foreground scene isolation:

| Scene | Bound Scale |
|-------|-------------|
| `ramen` | 1.5 |
| `figurines` | 0.025 |
| `teatime` | 1.0 |

**Mask-quality ablation** (Exps 1–6):

```bash
# ramen (bound scale = 1.5)
bash scripts/train_render_eval.sh ramen output_sam              object_mask_sam               normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam         crop_object_mask_sam          normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified      object_mask_sam_unified       normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified  normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_hq           object_mask_sam_hq            normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq      crop_object_mask_sam_hq       normal 1.5
```

**Loss-function ablation — Unified-Original baseline** (Exps 7–10):

> ⚠️ **Exps 9 and 10** use the VST loss or the combined loss (+Cont+Hyp+VST), which significantly increases memory consumption. To avoid GPU OOM, set `subdivide_until = 12000` and `prune_until = 15000` in the `src/config.py` file for the training of these experiments.

```bash
bash scripts/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp 1.5

# Exp 9 — VST only: use subdivide_until=12000 to avoid GPU memory overflow
bash scripts/train_render_eval.sh ramen output_sam_unified_vst object_mask_sam_unified vst 1.5

# Exp 10 — All losses (+Cont+Hyp+VST): use subdivide_until=12000 to avoid GPU memory overflow
bash scripts/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all 1.5
```

**Loss-function ablation — HQ-SAM Cropped baseline** (Exps 11–14):

> ⚠️ **Exps 13 and 14** follow the same memory constraint. Set `subdivide_until = 12000` and `prune_until = 15000` in the `src/config.py` file for the training of these experiments.

```bash
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_cos  crop_object_mask_sam_hq cos 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_hyp  crop_object_mask_sam_hq hyp 1.5

# Exp 13 — VST only: use subdivide_until=12000 to avoid GPU memory overflow
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_vst  crop_object_mask_sam_hq vst 1.5

# Exp 14 — All losses (+Cont+Hyp+VST): use subdivide_until=12000 to avoid GPU memory overflow
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_all  crop_object_mask_sam_hq all 1.5
```

> Repeat all commands with `figurines` (bound scale `0.025`) and `teatime` (bound scale `1`). See `runner_of_svraster.sh` for the full command listing.

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Iterations | 20,000 |
| Voxel pruning until | **15,000 iters** (vst / all — for Training Exps 9, 10, 13, 14) |
| Voxel pruning until | 18,000 iters (normal / cos / hyp — for all other Training Exps) |
| Voxel subdivision until | **12,000 iters** (vst / all — for Training Exps 9, 10, 13, 14) |
| Voxel subdivision until | 15,000 iters (normal / cos / hyp — for all other Training Exps) |
| num_objects (bits per dim) | 16 |
| num_classes | 256 |
| sh_objs_lr | 0.010 |
| λ²ᵈ | 0.05 |
| λ³ᵈ | 0.2 |
| λ_VST | 50 |
| λ²ᵈ_cont | 2×10⁻⁵ |
| λ³ᵈ_cont | 10⁻³ |
| λ²ᵈ_hyp | 5×10⁻⁵ |
| λ³ᵈ_hyp | 5×10⁻⁵ |
| VST update interval | 50 iters |
| VST views per update | 20 |

---

## Work 4 — Enhanced OpenGaussian

Work 4 builds on [OpenGaussian](https://github.com/muedavid/OpenGaussian) with a systematic evaluation of mask supervision quality and scene-cropping strategies for language-guided 3D segmentation. The two-phase training schedule (geometry + colour-features first, then language/object features) is retained from the original.

**Best configuration: Exp 7** — CAHMU-unified SAM on original images, no cropping → **59.02% mean mIoU** (new state-of-the-art on LeRF-OVS)

### Installation

> The conda environment for Work 4 is the same as Work 1 (`open_gaussian`). If Work 1 was already set up, **skip the installation steps** and proceed directly to the checkpoint setup and training.

```bash
cd modified-OpenGaussian

conda env create -f environment.yml
conda activate open_gaussian

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

pip install --no-build-isolation submodules/ashawkey-diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation submodules/sam-langsplat
pip install --no-build-isolation submodules/sam-hq

cd assets && unzip text_features.zip && cd ..
```

### Work 4 Checkpoint Setup

Work 4 requires a `ckpts/` folder containing two SAM model weights. Copy them from Work 2's (or Work 3's) `checkpoints/` folder:

```bash
# From the repository root
mkdir -p modified-OpenGaussian/ckpts

cp modified-gaussian-grouping/checkpoints/sam_hq_vit_h.pth \
   modified-OpenGaussian/ckpts/

cp modified-gaussian-grouping/checkpoints/sam_vit_h_4b8939.pth \
   modified-OpenGaussian/ckpts/
```

After this step the `ckpts/` folder should contain:

```
modified-OpenGaussian/ckpts/
├── sam_hq_vit_h.pth
└── sam_vit_h_4b8939.pth
```

### Step 1 — Generate all mask variants

Run the following preprocessing scripts to produce all six mask-type × crop-setting combinations for each scene:

```bash
# ramen
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/ramen
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/ramen          # CAHMU (Work 1)
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/ramen --crop

# figurines
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/figurines
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/figurines
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/figurines --crop

# teatime
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/teatime
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/teatime
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/teatime --crop
```

### Step 2 — Train the full-scene 3DGS geometry

```bash
python train_normal.py -s data/lerf_ovs/ramen     -m output_full_scene/ramen     --iterations 30_000
python train_normal.py -s data/lerf_ovs/figurines -m output_full_scene/figurines --iterations 30_000
python train_normal.py -s data/lerf_ovs/teatime   -m output_full_scene/teatime   --iterations 30_000
```

### Step 3 — (Optional) Crop scene for Crop@30k / Cropped-Render experiments

> Only needed for Exps 2, 3, 5, 6, 8, 9 (cropped-render and Crop@30k settings).

```bash
# ramen (padding = 1.5)
cp -r output_full_scene/ramen output/ramen
python crop_scene.py  -m output/ramen --iteration 30000 --padding 1.5
python crop_images.py -m output/ramen --iteration 30000

# figurines (padding = 0.025)
cp -r output_full_scene/figurines output/figurines
python crop_scene.py  -m output/figurines --iteration 30000 --padding 0.025
python crop_images.py -m output/figurines --iteration 30000

# teatime (padding = 1.0)
cp -r output_full_scene/teatime output/teatime
python crop_scene.py  -m output/teatime --iteration 30000 --padding 1
python crop_images.py -m output/teatime --iteration 30000
```

### Step 4 — Mask & cropping ablation (Exps 1–9)

All commands follow the pattern:

```bash
cp -r <geometry_source>/<scene> <output_dir>/<scene>
bash scripts/train_render_eval.sh <scene> <output_dir> <feature_dir> <mode> <checkpoint>
```

**ramen — nine configurations (repeat analogously for `figurines` and `teatime`):**

```bash
# Exp 1: Large SAM, original image
cp -r output_full_scene/ramen output_sam_l_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_l_full_scene language_features_sam_l normal chkpnt30000

# Exp 2: Large SAM, cropped render (no in-training crop)
cp -r output_full_scene/ramen output_crop_sam_l_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

# Exp 3: Large SAM, Crop@30k
cp -r output/ramen output_crop_sam_l/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

# Exp 4: HQ-SAM, original image
cp -r output_full_scene/ramen output_sam_hq_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

# Exp 5: HQ-SAM, cropped render
cp -r output_full_scene/ramen output_crop_sam_hq_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

# Exp 6: HQ-SAM, Crop@30k
cp -r output/ramen output_crop_sam_hq/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

# Exp 7: CAHMU-Unified SAM, original image  ← BEST overall configuration
cp -r output_full_scene/ramen output_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_u_full_scene language_features_sam_u normal chkpnt30000

# Exp 8: CAHMU-Unified SAM, cropped render
cp -r output_full_scene/ramen output_crop_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

# Exp 9: CAHMU-Unified SAM, Crop@30k  ← Operating baseline for ramen training/inference ablations
cp -r output/ramen output_crop_sam_u/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u crop_language_features_sam_u normal chkpnt30000
```

> Repeat the nine configurations for `figurines` and `teatime`. Padding values for `crop_scene.py`: `1.5` (ramen), `0.025` (figurines), `1.0` (teatime). See `runner_of_OpenGaussian.sh` for the complete listing.

### Step 5 — Training-method ablation (ramen only, Exps 11, 13, 15)

> Baseline is Exp 9 (Unified SAM, Crop@30k). Evaluated on ramen scene only; configurations observed to be clearly detrimental were not extended to other scenes.

```bash
# Exp 11: + Cluster Pruning (CP)
cp -r output_crop_sam_u/ramen output_crop_sam_u_prune/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_prune/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/ramen --split train

# Exp 13: + filter=False (NF)
cp -r output_crop_sam_u/ramen output_crop_sam_u_no_filter/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_no_filter/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/ramen --split train

# Exp 15: + CP + NF
cp -r output_crop_sam_u/ramen output_crop_sam_u_all/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_all/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/ramen --split train
```

### Step 6 — Inference re-ranking ablation (Exps 10, 12, 14, 16)

Inference re-ranking (two-stage: top-10 cosine → top-5 objectness re-rank) is invoked via `render_lerf_by_text.py` and `eval_lerf_mask_new.py` on top of each training configuration. Refer to the commented-out commands at the bottom of `runner_of_OpenGaussian.sh` for the full invocation sequence.

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Total iterations | 70,000 |
| Phase 1 — geometry + colour features | 1 – 30,000 |
| Phase 2 — object features | 30,001 – 70,000 |
| Cluster-pruning threshold θ_prune | 0.2 |
| Pruning interval | every 1,000 iters (Stage 2.1) |
| Re-ranking: cosine shortlist K₁ | 10 |
| Re-ranking: objectness shortlist K₂ | 5 |
| Feature-proximity threshold δ | 0.9 |
| Scene cropping multiplier β | 1.5 |

---

## Evaluation Metrics

All experiments are evaluated using a unified evaluation script reporting six complementary metrics:

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection-over-Union over all image–query pairs |
| **mBIoU** | Mean Boundary IoU (dilation ratio 0.02 × image diagonal) — boundary precision |
| **IoU Acc@0.25** | Fraction of query–image pairs with IoU > 0.25 |
| **IoU Acc@0.5** | Fraction of query–image pairs with IoU > 0.5 |
| **BIoU Acc@0.25** | Fraction of query–image pairs with BIoU > 0.25 |
| **BIoU Acc@0.5** | Fraction of query–image pairs with BIoU > 0.5 |

---

## Results

### Cross-Method Comparison (Best Configurations)

| Method | Mean mIoU ↑ | Mean mBIoU ↑ | IoU Acc@.25 ↑ | IoU Acc@.5 ↑ |
|--------|:-----------:|:------------:|:-------------:|:------------:|
| **Work 2** — Improved Gaussian Grouping (Exp 8) | 41.72 | 38.09 | 50.69 | 43.35 |
| **Work 3** — SVRaster + VST (Exp 13) | 45.53 | 42.65 | 54.81 | 47.18 |
| **Work 4** — Enhanced OpenGaussian (Exp 7) | **59.02** | **54.69** | **77.16** | **68.19** |

### Comparison with State-of-the-Art

| Method | Mean mIoU ↑ | Mean mBIoU ↑ | IoU Acc@.25 ↑ | IoU Acc@.5 ↑ |
|--------|:-----------:|:------------:|:-------------:|:------------:|
| LangSplat | 10.45 | 10.05 | 14.95 | 6.50 |
| LEGaussians | 17.85 | 16.90 | 26.65 | 11.45 |
| Gaussian Grouping | 36.04 | 33.40 | 44.46 | 36.65 |
| **Work 2 (Ours)** | 41.72 | 38.09 | 50.69 | 43.35 |
| **Work 3 (Ours)** | 45.53 | 42.65 | 54.81 | 47.18 |
| OpenGaussian | 53.77 | 49.89 | 69.54 | 59.01 |
| **Work 4 (Ours) ★** | **59.02** | **54.69** | **77.16** | **68.19** |

★ New state-of-the-art on LeRF-OVS. Work 4 exceeds the previous best (OpenGaussian) by **+5.25% mean mIoU** and **+9.18% IoU Acc@0.5**.

### Key Findings

> **Mask supervision quality is the dominant variable** — the largest single-step gains come from upgrading the mask type, not from any auxiliary loss, training modification, or inference change.

- **CAHMU-unified masks** are optimal for Gaussian-based representations (Works 2 and 4), delivering +5.08% / +5.25% over default / Large-level SAM respectively.
- **HQ-SAM on cropped renders** is optimal for the voxel backbone (Work 3), outperforming CAHMU-unified by +5.40% — the voxel depth-ordering amplifies HQ-SAM boundary precision.
- **In-training scene cropping (Crop@30k) is consistently detrimental** for OpenGaussian on LeRF tabletop scenes, degrading mean mIoU by ≈5–7%.
- Auxiliary losses (Cont, Hyp, GST/VST) produce effects within ≈1% mIoU on any baseline; **VST achieves the best mBIoU and IoU Acc@0.25** on Work 3's HQ-SAM voxel baseline.
- Objectness-based cluster pruning and two-stage re-ranking (Work 4) are counter-productive: they remove genuine foreground clusters or demote the correct cosine top-1 cluster.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@mastersthesis{mullick2025openvocab3d,
  title     = {Comparative Analysis of Open-Vocabulary 3D Scene Understanding
               using Enhanced SAM Masks and Multiple Segmentation Methods},
  author    = {Suvrajit Mullick},
  school    = {Indian Institute of Science (IISc), Bengaluru},
  year      = {2026},
  type      = {M.Tech Thesis Report, Department of Artificial Intelligence}
}
```

---

<div align="center">
<sub>Indian Institute of Science (IISc) · Bengaluru · M.Tech. Artificial Intelligence</sub>
</div>
