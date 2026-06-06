#!/usr/bin/env bash
# =============================================================================
#  Work 1 (CAHMU masks) + Work 4 — Enhanced OpenGaussian
#  Runner script: environment setup, checkpoint setup, mask preprocessing,
#  geometry training, and full experiment suite.
#  Run from the repository root, or place inside modified-OpenGaussian/ and
#  execute from there (adjust relative paths accordingly).
# =============================================================================

cd modified-OpenGaussian

# ── Environment ──────────────────────────────────────────────────────────────
conda env create -f environment.yml
conda activate open_gaussian

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# ── Submodule installation ────────────────────────────────────────────────────
pip install --no-build-isolation submodules/ashawkey-diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation submodules/sam-langsplat
pip install --no-build-isolation submodules/sam-hq

# ── Text features ─────────────────────────────────────────────────────────────
cd assets
unzip text_features.zip
cd ..

# ── Checkpoints — copy SAM weights from Work 2 (no re-download needed) ────────
# ⚠️  Complete Work 2 setup (runner_of_gaussian-grouping.sh) first so that
#     modified-gaussian-grouping/checkpoints/ is populated by download_models.sh.
mkdir -p ckpts
cp ../modified-gaussian-grouping/checkpoints/sam_hq_vit_h.pth     ckpts/
cp ../modified-gaussian-grouping/checkpoints/sam_vit_h_4b8939.pth ckpts/


# =============================================================================
#  Work 1 — CAHMU: generate unified single-level SAM masks (preprocess_sam_u.py)
#  These masks serve as supervision for Works 2 and 4.
# =============================================================================

# ramen
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/ramen
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/ramen          # CAHMU — Work 1 output
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/ramen --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_u

# figurines
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/figurines
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/figurines      # CAHMU — Work 1 output
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/figurines --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_u

# teatime
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/teatime
python preprocess_sam_l.py  --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/teatime        # CAHMU — Work 1 output
python preprocess_sam_u.py  --dataset_path data/lerf_ovs/teatime --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_u


# =============================================================================
#  Work 4 — Enhanced OpenGaussian
# =============================================================================

# ── Step 1: Full-scene 3DGS geometry training (30k iters) ────────────────────
python train_normal.py -s data/lerf_ovs/ramen     -m output_full_scene/ramen     --iterations 30_000
python train_normal.py -s data/lerf_ovs/figurines -m output_full_scene/figurines --iterations 30_000
python train_normal.py -s data/lerf_ovs/teatime   -m output_full_scene/teatime   --iterations 30_000

# ── Step 2: Scene cropping (for Exps 2, 3, 5, 6, 8, 9) ──────────────────────
# Produces the cropped geometry + cropped images used for Crop@30k and
# Cropped-Render configurations.  padding values are scene-specific.

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


# =============================================================================
#  RAMEN — Mask & cropping ablation (Exps 1–9)
# =============================================================================

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

# Exp 7: CAHMU-Unified SAM, original image  ← BEST overall configuration (59.02% mean mIoU)
cp -r output_full_scene/ramen output_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_u_full_scene language_features_sam_u normal chkpnt30000

# Exp 8: CAHMU-Unified SAM, cropped render
cp -r output_full_scene/ramen output_crop_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

# Exp 9: CAHMU-Unified SAM, Crop@30k  ← Operating baseline for ramen training/inference ablations
cp -r output/ramen output_crop_sam_u/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u crop_language_features_sam_u normal chkpnt30000

# ── RAMEN — Baseline render + eval (Exp 9) ───────────────────────────────────
python render_lerf_by_text.py -m output_crop_sam_u/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/ramen --split train

# ── RAMEN — Training-method ablation (Exps 11, 13, 15 — ramen only) ──────────

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

# ── RAMEN — Inference re-ranking ablation (Exps 10, 12, 14, 16 — ramen only) ─
# These apply two-stage objectness-based re-ranking on top of each training
# configuration above.  All were found to be detrimental; kept here for
# completeness and reproducibility.

# Exp 10: Exp 9 baseline + re-ranking
# python render_lerf_by_text.py -m output_crop_sam_u/ramen --scene_name ramen --skip_test --rerank
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/ramen --split train

# Exp 12: Exp 11 (CP) + re-ranking
# python render_lerf_by_text.py -m output_crop_sam_u_prune/ramen --scene_name ramen --skip_test --rerank
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/ramen --split train

# Exp 14: Exp 13 (NF) + re-ranking
# python render_lerf_by_text.py -m output_crop_sam_u_no_filter/ramen --scene_name ramen --skip_test --rerank
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/ramen --split train

# Exp 16: Exp 15 (CP + NF) + re-ranking
# python render_lerf_by_text.py -m output_crop_sam_u_all/ramen --scene_name ramen --skip_test --rerank
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/ramen --split train


# =============================================================================
#  FIGURINES — Mask & cropping ablation (Exps 1–9)
# =============================================================================

# Exp 1: Large SAM, original image
cp -r output_full_scene/figurines output_sam_l_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_l_full_scene language_features_sam_l normal chkpnt30000

# Exp 2: Large SAM, cropped render
cp -r output_full_scene/figurines output_crop_sam_l_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

# Exp 3: Large SAM, Crop@30k
cp -r output/figurines output_crop_sam_l/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

# Exp 4: HQ-SAM, original image
cp -r output_full_scene/figurines output_sam_hq_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

# Exp 5: HQ-SAM, cropped render
cp -r output_full_scene/figurines output_crop_sam_hq_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

# Exp 6: HQ-SAM, Crop@30k
cp -r output/figurines output_crop_sam_hq/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

# Exp 7: CAHMU-Unified SAM, original image  ← BEST overall configuration
cp -r output_full_scene/figurines output_sam_u_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_u_full_scene language_features_sam_u normal chkpnt30000

# Exp 8: CAHMU-Unified SAM, cropped render
cp -r output_full_scene/figurines output_crop_sam_u_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

# Exp 9: CAHMU-Unified SAM, Crop@30k
cp -r output/figurines output_crop_sam_u/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_u crop_language_features_sam_u normal chkpnt30000

# Training-method and re-ranking ablations are ramen-only — not run for figurines.
# python render_lerf_by_text.py -m output_crop_sam_u/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_prune/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_prune/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_no_filter/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_no_filter/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_all/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_all/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/figurines --split train


# =============================================================================
#  TEATIME — Mask & cropping ablation (Exps 1–9)
# =============================================================================

# Exp 1: Large SAM, original image
cp -r output_full_scene/teatime output_sam_l_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_l_full_scene language_features_sam_l normal chkpnt30000

# Exp 2: Large SAM, cropped render
cp -r output_full_scene/teatime output_crop_sam_l_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

# Exp 3: Large SAM, Crop@30k
cp -r output/teatime output_crop_sam_l/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

# Exp 4: HQ-SAM, original image
cp -r output_full_scene/teatime output_sam_hq_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

# Exp 5: HQ-SAM, cropped render
cp -r output_full_scene/teatime output_crop_sam_hq_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

# Exp 6: HQ-SAM, Crop@30k
cp -r output/teatime output_crop_sam_hq/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

# Exp 7: CAHMU-Unified SAM, original image  ← BEST overall configuration
cp -r output_full_scene/teatime output_sam_u_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_u_full_scene language_features_sam_u normal chkpnt30000

# Exp 8: CAHMU-Unified SAM, cropped render
cp -r output_full_scene/teatime output_crop_sam_u_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

# Exp 9: CAHMU-Unified SAM, Crop@30k
cp -r output/teatime output_crop_sam_u/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_u crop_language_features_sam_u normal chkpnt30000

# Training-method and re-ranking ablations are ramen-only — not run for teatime.
# python render_lerf_by_text.py -m output_crop_sam_u/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_prune/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_prune/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_no_filter/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_no_filter/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_all/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_all/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/teatime --split train