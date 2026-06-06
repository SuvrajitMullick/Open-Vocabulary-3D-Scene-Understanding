#!/bin/bash


# Check if the user provided an argument
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"
output_name="$2"
masks_name="$3"
train_mode="$4"


# Modified Gaussian Grouping training
python train_${train_mode}.py \
    -s data/${dataset_name} \
    -r 1 \
    -m ${output_name}/${dataset_name} \
    --object_path ${masks_name} \
    --config_file config/gaussian_dataset/train.json \
    --optimizer_type sparse_adam \
    -d depth_maps \
    --antialiasing \
    --disable_viewer

# Segmentation rendering by text queries using trained model
python render_lerf_by_text.py -m ${output_name}/${dataset_name} --skip_test

# Evaluation of Segmentation Masks
python script/eval_lerf_mask.py --out_dir ${output_name}/${dataset_name} --split train


# bash script/train_render_eval.sh ramen output_sam object_mask_sam normal