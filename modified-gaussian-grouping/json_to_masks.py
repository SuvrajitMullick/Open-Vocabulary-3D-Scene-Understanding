import json
import os
import argparse
import glob
from PIL import Image, ImageDraw

def generate_category_masks_from_json(json_path, output_base_dir, frame_name):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    width = data['info']['width']
    height = data['info']['height']

    # Locate the corresponding original image
    image_path = json_path.replace(".json", ".jpg")
    original_image = None
    blank_image = None
    if os.path.exists(image_path):
        # Convert to RGBA to support a transparent background for the segmented objects
        original_image = Image.open(image_path).convert("RGBA")
        blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        print(f"Warning: Original image not found for {frame_name}. Only masks will be generated.")

    # Create subdirectories for masks and rgbs
    mask_dir = os.path.join(output_base_dir, frame_name, "masks")
    rgb_dir = os.path.join(output_base_dir, frame_name, "rgbs")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    
    category_images = {}
    category_draws = {}

    # Draw the polygons
    for obj in data['objects']:
        category = obj['category'].replace(" ", "_")

        if category not in category_images:
            mask_image = Image.new('L', (width, height), color=0)
            category_images[category] = mask_image
            category_draws[category] = ImageDraw.Draw(mask_image)

        polygon_points = [tuple(point) for point in obj['segmentation']]
        category_draws[category].polygon(polygon_points, outline=255, fill=255)

    # Save outputs
    for category, mask_image in category_images.items():
        # 1. Save the binary mask
        mask_file_name = f"{category}.jpg"
        mask_save_path = os.path.join(mask_dir, mask_file_name)
        mask_image.save(mask_save_path)

        # 2. Save the segmented RGB (if original image was found)
        if original_image:
            # Composite uses the mask to decide which pixels to keep from original vs blank image
            segmented_rgb = Image.composite(original_image, blank_image, mask_image)
            
            # Save as PNG to preserve the transparent background
            rgb_file_name = f"{category}.png"
            rgb_save_path = os.path.join(rgb_dir, rgb_file_name)
            segmented_rgb.save(rgb_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing the JSON files.")

    args = parser.parse_args()
    if not args.data_dir:
        parser.error("The --data_dir argument is required.")

    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files. Processing...")
    
    for json_path in json_files:
        base_name = os.path.basename(json_path)
        frame_name = os.path.splitext(base_name)[0]
        generate_category_masks_from_json(json_path, args.data_dir, frame_name)
        
    print("Done!")