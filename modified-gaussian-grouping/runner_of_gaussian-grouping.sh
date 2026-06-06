# Create environment
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


# Install DEVA, SAM, SAM-HQ & Depth-Anything models
bash script/download_models.sh

wget -P config/ https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py


python json_to_masks.py --data_dir data/label/ramen
python json_to_masks.py --data_dir data/label/figurines
python json_to_masks.py --data_dir data/label/teatime



# Prepare the Masks
bash script/prepare_pseudo_label.sh ramen 1 sam
bash script/prepare_pseudo_label.sh ramen 1 sam_hq
bash script/prepare_pseudo_label.sh ramen 1 sam_unified
bash script/prepare_pseudo_label.sh ramen crop sam
bash script/prepare_pseudo_label.sh ramen crop sam_hq
bash script/prepare_pseudo_label.sh ramen crop sam_unified

bash script/prepare_pseudo_label.sh figurines 1 sam
bash script/prepare_pseudo_label.sh figurines 1 sam_hq
bash script/prepare_pseudo_label.sh figurines 1 sam_unified
bash script/prepare_pseudo_label.sh figurines crop sam
bash script/prepare_pseudo_label.sh figurines crop sam_hq
bash script/prepare_pseudo_label.sh figurines crop sam_unified

bash script/prepare_pseudo_label.sh teatime 1 sam
bash script/prepare_pseudo_label.sh teatime 1 sam_hq
bash script/prepare_pseudo_label.sh teatime 1 sam_unified
bash script/prepare_pseudo_label.sh teatime crop sam
bash script/prepare_pseudo_label.sh teatime crop sam_hq
bash script/prepare_pseudo_label.sh teatime crop sam_unified




bash script/train_render_eval.sh ramen output_sam object_mask_sam normal
bash script/train_render_eval.sh ramen output_crop_sam crop_object_mask_sam normal
bash script/train_render_eval.sh ramen output_sam_unified object_mask_sam_unified normal
bash script/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh ramen output_sam_hq object_mask_sam_hq normal
bash script/train_render_eval.sh ramen output_crop_sam_hq crop_object_mask_sam_hq normal


bash script/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh ramen output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all


bash script/train_render_eval.sh ramen output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh ramen output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh ramen output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh ramen output_crop_sam_hq_all crop_object_mask_sam_hq all




bash script/train_render_eval.sh figurines output_sam object_mask_sam normal
bash script/train_render_eval.sh figurines output_crop_sam crop_object_mask_sam normal
bash script/train_render_eval.sh figurines output_sam_unified object_mask_sam_unified normal
bash script/train_render_eval.sh figurines output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh figurines output_sam_hq object_mask_sam_hq normal
bash script/train_render_eval.sh figurines output_crop_sam_hq crop_object_mask_sam_hq normal


bash script/train_render_eval.sh figurines output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh figurines output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh figurines output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh figurines output_sam_unified_all object_mask_sam_unified all


bash script/train_render_eval.sh figurines output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh figurines output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh figurines output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh figurines output_crop_sam_hq_all crop_object_mask_sam_hq all




bash script/train_render_eval.sh teatime output_sam object_mask_sam normal
bash script/train_render_eval.sh teatime output_crop_sam crop_object_mask_sam normal
bash script/train_render_eval.sh teatime output_sam_unified object_mask_sam_unified normal
bash script/train_render_eval.sh teatime output_crop_sam_unified crop_object_mask_sam_unified normal
bash script/train_render_eval.sh teatime output_sam_hq object_mask_sam_hq normal
bash script/train_render_eval.sh teatime output_crop_sam_hq crop_object_mask_sam_hq normal


bash script/train_render_eval.sh teatime output_sam_unified_cos object_mask_sam_unified cos
bash script/train_render_eval.sh teatime output_sam_unified_hyp object_mask_sam_unified hyp
bash script/train_render_eval.sh teatime output_sam_unified_gst object_mask_sam_unified gst
bash script/train_render_eval.sh teatime output_sam_unified_all object_mask_sam_unified all


bash script/train_render_eval.sh teatime output_crop_sam_hq_cos crop_object_mask_sam_hq cos
bash script/train_render_eval.sh teatime output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp
bash script/train_render_eval.sh teatime output_crop_sam_hq_gst crop_object_mask_sam_hq gst
bash script/train_render_eval.sh teatime output_crop_sam_hq_all crop_object_mask_sam_hq all


