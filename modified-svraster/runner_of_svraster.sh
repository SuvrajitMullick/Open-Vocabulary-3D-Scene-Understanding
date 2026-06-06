#!/usr/bin/env bash
# =============================================================================
#  Work 3 — Segmentation over Sparse Voxel Rasterization (SVRaster)
#  Runner script: environment setup, data/checkpoint reuse from Work 2,
#  and full experiment suite.
#  Run from the repository root, or place inside modified-svraster/ and
#  execute from there (adjust relative paths accordingly).
# =============================================================================

cd modified-svraster

# ── Environment ──────────────────────────────────────────────────────────────
conda env create -f environment.yml
conda activate svraster

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# ── Submodule installation ────────────────────────────────────────────────────
pip install --no-build-isolation ./cuda/
pip install --no-build-isolation ./fused-ssim/
pip install --no-build-isolation ./GroundingDINO/
pip install --no-build-isolation ./sam-hq/

# ── GroundingDINO config ──────────────────────────────────────────────────────
# Copy directly from Work 2's config/ folder (already downloaded manually there).
# ⚠️  Work 3 expects the file under cfg/ (not config/).
cp ../modified-gaussian-grouping/config/GroundingDINO_SwinB_cfg.py cfg/

# ── Checkpoints — copy from Work 2 (no re-download needed) ───────────────────
mkdir -p checkpoints
cp ../modified-gaussian-grouping/checkpoints/groundingdino_swinb_cogcoor.pth checkpoints/
cp ../modified-gaussian-grouping/checkpoints/sam_hq_vit_h.pth                checkpoints/
cp ../modified-gaussian-grouping/checkpoints/sam_vit_h_4b8939.pth            checkpoints/

# ── Data — copy preprocessed data/ from Work 2 (no re-preprocessing needed) ──
# Work 3 uses the same scene images and the same prepared mask folders as Work 2.
# Complete the full Work 2 pseudo-label preparation (runner_of_gaussian-grouping.sh)
# before running this step.
cp -r ../modified-gaussian-grouping/data data


# =============================================================================
#  RAMEN (bound_scale = 1.5)
# =============================================================================

# ── Mask-quality ablation (Exps 1–6) ─────────────────────────────────────────
bash scripts/train_render_eval.sh ramen output_sam              object_mask_sam              normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam         crop_object_mask_sam         normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified      object_mask_sam_unified      normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_hq           object_mask_sam_hq           normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq      crop_object_mask_sam_hq      normal 1.5

# ── Loss-function ablation: Unified-Original baseline (Exps 7–10) ────────────
bash scripts/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp 1.5

# Exp 9 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh ramen output_sam_unified_vst object_mask_sam_unified vst 1.5

# Exp 10 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all 1.5

# ── Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14) ─────────────
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_cos crop_object_mask_sam_hq cos 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 1.5

# Exp 13 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_vst crop_object_mask_sam_hq vst 1.5

# Exp 14 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_all crop_object_mask_sam_hq all 1.5


# =============================================================================
#  FIGURINES (bound_scale = 0.025)
# =============================================================================

# ── Mask-quality ablation (Exps 1–6) ─────────────────────────────────────────
bash scripts/train_render_eval.sh figurines output_sam              object_mask_sam              normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam         crop_object_mask_sam         normal 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified      object_mask_sam_unified      normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_unified crop_object_mask_sam_unified normal 0.025
bash scripts/train_render_eval.sh figurines output_sam_hq           object_mask_sam_hq           normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq      crop_object_mask_sam_hq      normal 0.025

# ── Loss-function ablation: Unified-Original baseline (Exps 7–10) ────────────
bash scripts/train_render_eval.sh figurines output_sam_unified_cos object_mask_sam_unified cos 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified_hyp object_mask_sam_unified hyp 0.025

# Exp 9 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh figurines output_sam_unified_vst object_mask_sam_unified vst 0.025

# Exp 10 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh figurines output_sam_unified_all object_mask_sam_unified all 0.025

# ── Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14) ─────────────
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_cos crop_object_mask_sam_hq cos 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 0.025

# Exp 13 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_vst crop_object_mask_sam_hq vst 0.025

# Exp 14 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_all crop_object_mask_sam_hq all 0.025


# =============================================================================
#  TEATIME (bound_scale = 1)
# =============================================================================

# ── Mask-quality ablation (Exps 1–6) ─────────────────────────────────────────
bash scripts/train_render_eval.sh teatime output_sam              object_mask_sam              normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam         crop_object_mask_sam         normal 1
bash scripts/train_render_eval.sh teatime output_sam_unified      object_mask_sam_unified      normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam_unified crop_object_mask_sam_unified normal 1
bash scripts/train_render_eval.sh teatime output_sam_hq           object_mask_sam_hq           normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq      crop_object_mask_sam_hq      normal 1

# ── Loss-function ablation: Unified-Original baseline (Exps 7–10) ────────────
bash scripts/train_render_eval.sh teatime output_sam_unified_cos object_mask_sam_unified cos 1
bash scripts/train_render_eval.sh teatime output_sam_unified_hyp object_mask_sam_unified hyp 1

# Exp 9 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh teatime output_sam_unified_vst object_mask_sam_unified vst 1

# Exp 10 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh teatime output_sam_unified_all object_mask_sam_unified all 1

# ── Loss-function ablation: HQ-SAM Cropped baseline (Exps 11–14) ─────────────
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_cos crop_object_mask_sam_hq cos 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 1

# Exp 13 — VST only
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_vst crop_object_mask_sam_hq vst 1

# Exp 14 — All losses (+Cont +Hyp +VST)
# ⚠️  GPU memory: set subdivide_until=12000 and prune_until=15000 in src/config.py
#     before running this experiment to avoid OOM.
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_all crop_object_mask_sam_hq all 1