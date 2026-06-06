#!/usr/bin/env bash
# =============================================================================
#  Work 2 — Improved Gaussian Grouping
#  Runner script: environment setup, data preprocessing, and full experiment suite
#  Run from the repository root, or place inside modified-gaussian-grouping/ and
#  execute from there.
# =============================================================================

cd modified-gaussian-grouping

# ── Environment ──────────────────────────────────────────────────────────────
conda env create -f environment.yml
conda activate my_gaussian_grouping

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# ── Submodule installation ────────────────────────────────────────────────────
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/fused-ssim
pip install --no-build-isolation ./submodules/GroundingDINO
pip install --no-build-isolation ./submodules/deva
pip install --no-build-isolation ./submodules/sam-hq
pip install --no-build-isolation ./submodules/depth-anything-v2

# ── Model checkpoints (DEVA, SAM, SAM-HQ, Depth-Anything) ────────────────────
# Populates the checkpoints/ folder with all required .pth files.
bash script/download_models.sh

# ── GroundingDINO config ──────────────────────────────────────────────────────
# ⚠️  Do NOT use wget — it fetches the HTML page, not the raw Python file.
#     Download the raw file manually:
#       1. Open the URL below in a browser.
#       2. Click the "Raw" button.
#       3. Save the file as  GroundingDINO_SwinB_cfg.py
#       4. Move it to  config/GroundingDINO_SwinB_cfg.py
#
#  URL:
#  https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py

# ── JSON labels → binary masks ───────────────────────────────────────────────
python json_to_masks.py --data_dir data/label/ramen
python json_to_masks.py --data_dir data/label/figurines
python json_to_masks.py --data_dir data/label/teatime

# ── Pseudo-label preparation (mask supervision) ───────────────────────────────
# Three mask types (sam / sam_hq / sam_unified) × two image sources (1 = original, crop = cropped render)
# Produces all mask folders needed for the 6-configuration mask-quality ablation (Exps 1–6).
# These masks are also reused by Work 3 — see runner_of_svraster.sh for the cp command.

# ramen
bash script/prepare_pseudo_label.sh ramen 1    sam
bash script/prepare_pseudo_label.sh ramen 1    sam_hq
bash script/prepare_pseudo_label.sh ramen 1    sam_unified
bash script/prepare_pseudo_label.sh ramen crop sam
bash script/prepare_pseudo_label.sh ramen crop sam_hq
bash script/prepare_pseudo_label.sh ramen crop sam_unified

# figurines
bash script/prepare_pseudo_label.sh figurines 1    sam
bash script/prepare_pseudo_label.sh figurines 1    sam_hq
bash script/prepare_pseudo_label.sh figurines 1    sam_unified
bash script/prepare_pseudo_label.sh figurines crop sam
bash script/prepare_pseudo_label.sh figurines crop sam_hq
bash script/prepare_pseudo_label.sh figurines crop sam_unified

# teatime
bash script/prepare_pseudo_label.sh teatime 1    sam
bash script/prepare_pseudo_label.sh teatime 1    sam_hq
bash script/prepare_pseudo_label.sh teatime 1    sam_unified
bash script/prepare_pseudo_label.sh teatime crop sam
bash script/prepare_pseudo_label.sh teatime crop sam_hq
bash script/prepare_pseudo_label.sh teatime crop sam_unified


# =============================================================================
#  RAMEN — Mask-quality ablation (Exps 1–6)
# =============================================================================
bash script/train_render_eval.sh ramen output_sam              object_mask_sam              normal
bash script/train_render_eval.sh ramen output_crop_sam         crop_object_mask_sam         normal
bash script/train_render_eval.sh ramen output_sam_unified      object_mask_sam_unified      normal
bash script/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh ramen output_sam_hq           object_mask_sam_hq           normal
bash script/train_render_eval.sh ramen output_crop_sam_hq      crop_object_mask_sam_hq      normal

# RAMEN — Loss-function ablation: Unified-Original baseline (Exps 7–10)
bash script/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh ramen output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all

# RAMEN — Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14)
bash script/train_render_eval.sh ramen output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh ramen output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh ramen output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh ramen output_crop_sam_hq_all crop_object_mask_sam_hq all


# =============================================================================
#  FIGURINES — Mask-quality ablation (Exps 1–6)
# =============================================================================
bash script/train_render_eval.sh figurines output_sam              object_mask_sam              normal
bash script/train_render_eval.sh figurines output_crop_sam         crop_object_mask_sam         normal
bash script/train_render_eval.sh figurines output_sam_unified      object_mask_sam_unified      normal
bash script/train_render_eval.sh figurines output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh figurines output_sam_hq           object_mask_sam_hq           normal
bash script/train_render_eval.sh figurines output_crop_sam_hq      crop_object_mask_sam_hq      normal

# FIGURINES — Loss-function ablation: Unified-Original baseline (Exps 7–10)
bash script/train_render_eval.sh figurines output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh figurines output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh figurines output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh figurines output_sam_unified_all object_mask_sam_unified all

# FIGURINES — Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14)
bash script/train_render_eval.sh figurines output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh figurines output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh figurines output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh figurines output_crop_sam_hq_all crop_object_mask_sam_hq all


# =============================================================================
#  TEATIME — Mask-quality ablation (Exps 1–6)
# =============================================================================
bash script/train_render_eval.sh teatime output_sam              object_mask_sam              normal
bash script/train_render_eval.sh teatime output_crop_sam         crop_object_mask_sam         normal
bash script/train_render_eval.sh teatime output_sam_unified      object_mask_sam_unified      normal
bash script/train_render_eval.sh teatime output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh teatime output_sam_hq           object_mask_sam_hq           normal
bash script/train_render_eval.sh teatime output_crop_sam_hq      crop_object_mask_sam_hq      normal

# TEATIME — Loss-function ablation: Unified-Original baseline (Exps 7–10)
bash script/train_render_eval.sh teatime output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh teatime output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh teatime output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh teatime output_sam_unified_all object_mask_sam_unified all

# TEATIME — Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14)
bash script/train_render_eval.sh teatime output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh teatime output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh teatime output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh teatime output_crop_sam_hq_all crop_object_mask_sam_hq all