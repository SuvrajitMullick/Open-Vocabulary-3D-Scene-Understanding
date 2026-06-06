import os
import glob
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def visualize_new_masks(root_dir, crop, model_name):
    """
    Reads new _s.npy SAM mask files from root_dir/${prefix}language_features_${model_name} 
    and saves visualizations to root_dir/${prefix}masks_visualization_${model_name}.
    """
    prefix = "crop_" if crop else ""
    features_folder = os.path.join(root_dir, prefix + 'language_features_' + model_name)
    output_folder = os.path.join(root_dir, prefix + 'masks_visualization_' + model_name)

    if not os.path.exists(features_folder):
        print(f"[ ERROR ] The directory '{features_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    npy_files = glob.glob(os.path.join(features_folder, "*_s.npy"))
    npy_files.sort()

    if not npy_files:
        print(f"[ ERROR ] No '_s.npy' files found in {features_folder}")
        return

    print(f"Found {len(npy_files)} mask files in {features_folder}.")
    print(f"Generating visualizations and saving to {output_folder}...")

    np.random.seed(42)
    color_palette = np.random.randint(50, 256, size=(1000, 3), dtype=np.uint8)

    for npy_path in tqdm(npy_files, desc="Visualizing masks"):
        seg_map_tensor = np.load(npy_path)
        
        # Extract the new layer (index 0 if 3D tensor)
        if seg_map_tensor.ndim == 3:
            seg_map = seg_map_tensor[0]
        else:
            seg_map = seg_map_tensor

        h, w = seg_map.shape
        vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        valid_pixels = seg_map >= 0
        
        if valid_pixels.any():
            vis_image[valid_pixels] = color_palette[seg_map[valid_pixels] % len(color_palette)]

        base_name = os.path.basename(npy_path).replace('_s.npy', '_mask.png')
        save_path = os.path.join(output_folder, base_name)

        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, vis_image_bgr)

    print(f"\n[ SUCCESS ] All new visualizations saved to: {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize new SAM masks from the SHPA pipeline.")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--crop', action='store_true', help='Enable cropping')
    parser.add_argument('--variant', type=str, required=True)
    args = parser.parse_args()
    visualize_new_masks(args.dataset_path, args.crop, args.variant)