#!/bin/bash


# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi


dataset_name="$1"
scale="$2"
model_name="$3"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# 1. DEVA anything mask
if [ "$scale" = "1" ]; then
    img_path="${dataset_folder}/images"
    out_path="mask_output/${dataset_name}_${model_name}"
    object_folder="${dataset_folder}/object_mask_${model_name}"
elif [ "$scale" = "crop" ]; then
    img_path="${dataset_folder}/crop_images"
    out_path="mask_output/crop_${dataset_name}_${model_name}"
    object_folder="${dataset_folder}/crop_object_mask_${model_name}"
else
    img_path="${dataset_folder}/images_${scale}"
    out_path="mask_output/${dataset_name}_${scale}_${model_name}"
    object_folder="${dataset_folder}/object_mask_${scale}_${model_name}"
fi


# colored mask for visualization check
python script/deva_automatic_mask.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "$out_path" \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --suppress_small_objects  \
  --sam_variant "$model_name" \


mv ${out_path}/Annotations ${out_path}/Annotations_color

# gray mask for training
python script/deva_automatic_mask.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "$out_path" \
  --use_short_id  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --suppress_small_objects  \
  --sam_variant "$model_name" \


# 2. copy gray mask to the correponding data path
cp -r ${out_path}/Annotations $object_folder

python script/depth_maps_run.py --encoder vitg --pred-only --grayscale --img-path "$img_path" --outdir ${dataset_folder}/depth_maps

python utils/make_depth_scale.py --base_dir $dataset_folder --depths_dir ${dataset_folder}/depth_maps

