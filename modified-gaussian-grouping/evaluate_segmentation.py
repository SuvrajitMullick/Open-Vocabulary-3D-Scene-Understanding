#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

from pathlib import Path
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
from argparse import ArgumentParser

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Converts a binary mask to a boundary mask for BIoU.
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    num_pixels = int(img_diag * dilation_ratio)
    if num_pixels < 1: num_pixels = 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (num_pixels, num_pixels))
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded_mask = cv2.erode(padded_mask, kernel, iterations=1)
    boundary = padded_mask - eroded_mask
    return boundary[1:-1, 1:-1]

def get_unique_colors(image):
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return set(map(tuple, unique_colors))

def get_binary_mask(image, color):
    mask = np.all(image == color, axis=-1)
    return mask.astype(np.uint8)

def compute_metrics_for_image(gt_path, render_path):
    """
    Calculates mIoU and mBIoU for a single pair of images.
    """
    # Load images using OpenCV (BGR) and convert to RGB
    gt_img = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
    pred_img = cv2.cvtColor(cv2.imread(str(render_path)), cv2.COLOR_BGR2RGB)

    # Resize pred to match GT if dimensions differ
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Identify all Classes (Unique Colors) in this specific view
    gt_colors = get_unique_colors(gt_img)
    pred_colors = get_unique_colors(pred_img)
    all_colors = sorted(list(gt_colors | pred_colors)) 

    iou_scores = []
    biou_scores = []

    for color in all_colors:
        gt_binary = get_binary_mask(gt_img, color)
        pred_binary = get_binary_mask(pred_img, color)

        # --- IoU ---
        intersection = np.sum(np.logical_and(gt_binary, pred_binary))
        union = np.sum(np.logical_or(gt_binary, pred_binary))
        
        if union > 0:
            iou_scores.append(intersection / union)

        # --- BIoU ---
        # Only calculate if the class is present in at least one
        if np.sum(gt_binary) > 0 or np.sum(pred_binary) > 0:
            gt_boundary = mask_to_boundary(gt_binary)
            pred_boundary = mask_to_boundary(pred_binary)
            
            b_intersection = np.sum(np.logical_and(gt_boundary, pred_boundary))
            b_union = np.sum(np.logical_or(gt_boundary, pred_boundary))
            
            if b_union == 0:
                biou_scores.append(0.0)
            else:
                biou_scores.append(b_intersection / b_union)

    # Average over classes for this image
    mIoU = np.mean(iou_scores) if iou_scores else 0.0
    mBIoU = np.mean(biou_scores) if biou_scores else 0.0
    
    return mIoU, mBIoU

def readImagePaths(renders_dir, gt_dir):
    # Modified to return paths instead of loading all tensors to memory immediately
    # This prevents OOM errors with large segmentation maps
    render_paths = []
    gt_paths = []
    image_names = []
    
    # Filter for image extensions
    valid_exts = {'.png', '.jpg', '.jpeg'}
    
    for fname in os.listdir(renders_dir):
        if Path(fname).suffix.lower() in valid_exts:
            render_paths.append(renders_dir / fname)
            gt_paths.append(gt_dir / fname)
            image_names.append(fname)
            
    return render_paths, gt_paths, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "train"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt_objects_color"
                renders_dir = method_dir / "objects_pred"
                
                # Check if directories exist
                if not os.path.exists(renders_dir) or not os.path.exists(gt_dir):
                    print(f"Skipping {method}: 'renders' or 'gt' folder missing.")
                    continue

                render_paths, gt_paths, image_names = readImagePaths(renders_dir, gt_dir)

                mious = []
                mbious = []

                print(f"Evaluating {len(render_paths)} images...")

                for idx in tqdm(range(len(render_paths)), desc="Metric evaluation progress"):
                    # Calculate metrics per image
                    miou_val, mbiou_val = compute_metrics_for_image(gt_paths[idx], render_paths[idx])
                    
                    mious.append(miou_val)
                    mbious.append(mbiou_val)

                # Averages across the dataset
                mean_miou = np.mean(mious)
                mean_mbiou = np.mean(mbious)

                print("  mIoU  : {:>12.7f}".format(mean_miou))
                print("  mBIoU : {:>12.7f}".format(mean_mbiou))
                print("")

                full_dict[scene_dir][method].update({
                    "mIoU": mean_miou,
                    "mBIoU": mean_mbiou
                })
                
                per_view_dict[scene_dir][method].update({
                    "mIoU": {name: val for val, name in zip(mious, image_names)},
                    "mBIoU": {name: val for val, name in zip(mbious, image_names)}
                })

            with open(scene_dir + "/results_segmentation.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_segmentation.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Error:", e)

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Segmentation Evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    
    evaluate(args.model_paths)