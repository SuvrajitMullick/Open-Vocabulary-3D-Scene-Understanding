#!/bin/bash


# Check if the user provided an argument
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"
output_name="$2"
masks_name="$3"
train_mode="$4"
scale="$5"


# Modified Sparse Voxels Rasterization with Voxels Grouping training
python train_${train_mode}.py \
    --source_path data/${dataset_name} \
    --model_path ${output_name}/${dataset_name} \
    --image_dir_name images \
    --object_dir_name ${masks_name} \
    --lambda_normal_dmean 0.02 \
    --lambda_normal_dmed 0.02 \
    --lambda_T_inside 0.2 \
    --lambda_T_concen 2.0 \
    --bound_mode camera_max \
    --bound_scale ${scale} \
    --res_downscale 1

# Segmentation rendering by text queries using trained model
python render_lerf_by_text.py --model_path ${output_name}/${dataset_name} --skip_test

# Evaluation of Segmentation Masks
python scripts/eval_lerf_mask.py --out_dir ${output_name}/${dataset_name} --split train


# bash script/train_render_eval.sh ramen output_sam object_mask_sam normal