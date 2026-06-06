#!/bin/bash


# Check if the user provided an argument
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"
output_name="$2"
features_name="$3"
train_mode="$4"
chkpnt_name="$5"


# Modified Open-Gaussian training
python train_${train_mode}.py \
    -s data/lerf_ovs/${dataset_name} \
    -m ${output_name}/${dataset_name} \
    --features ${features_name} \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --root_node_num 64 \
    --leaf_node_num 10 \
    --pos_weight 0.5 \
    --loss_weight 0.01 \
    --test_iterations 30000 \
    --start_checkpoint ${output_name}/${dataset_name}/${chkpnt_name}.pth

# Segmentation rendering by text queries using trained model
python render_lerf_by_text_normal.py -m ${output_name}/${dataset_name} --scene_name ${dataset_name} --skip_test

# Evaluation of Segmentation Masks
python scripts/eval_lerf_mask.py --out_dir ${output_name}/${dataset_name} --split train


# bash script/train_render_eval.sh ramen output_sam_l_full_scene language_features_sam_l normal chkpnt30000