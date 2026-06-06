import os
import numpy as np
from PIL import Image
import cv2
import json
from argparse import ArgumentParser

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02, base=None):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :param base (str): "former", "later", or None (for union)
    :return: boundary iou (float)
    """
    dt = (dt>128).astype('uint8')
    gt = (gt>128).astype('uint8')

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    
    if base == "former":
        union = (gt_boundary > 0).sum()
    elif base == "later":
        union = (dt_boundary > 0).sum()
    else:
        union = ((gt_boundary + dt_boundary) > 0).sum()
    
    if union == 0:
        return 0.0
    boundary_iou = intersection / union
    return boundary_iou


def load_mask(mask_path):
    """Load the mask from the given path."""
    if os.path.exists(mask_path):
        return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
    return None

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def calculate_iou(mask1, mask2, base=None):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    
    if base == "former":
        union = mask1_bool.sum()
    elif base == "later":
        union = mask2_bool.sum()
    else:
        union = np.logical_or(mask1_bool, mask2_bool).sum()
    
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def evaluate(out_dir, base_mode, split, iteration):
    gt_path = os.path.join('data/lerf_ovs/label', os.path.basename(out_dir))
    pred_path = os.path.join(out_dir, 'text2obj_' + split + '_new', 'ours_' + iteration, 'renders_cluster_silhouette')
    
    iou_scores = {}  # Store IoU scores for each class
    biou_scores = {} # Store BIoU scores for each class
    all_ious = []    # Store all flat IoU scores for overall accuracy
    all_bious = []   # Store all flat BIoU scores for overall accuracy

    # Data structures for JSON output
    full_dict = {}
    per_view_dict = {}

    # Iterate over each image and category in the GT dataset
    for image_name in os.listdir(gt_path):
        gt_image_path = os.path.join(gt_path, image_name, 'masks')
        pred_image_path = os.path.join(pred_path, image_name)

        if os.path.isdir(gt_image_path):
            per_view_dict[image_name] = {}

            for cat_file in os.listdir(gt_image_path):
                cat_id = cat_file.split('.')[0]  # Assuming cat_file format is "cat_id.png"
                gt_mask_path = os.path.join(gt_image_path, cat_file)
                pred_mask_path = os.path.join(pred_image_path, f"{cat_id}.png")

                gt_mask = load_mask(gt_mask_path)
                pred_mask = load_mask(pred_mask_path)
                # print("GT:  ",gt_mask_path)
                # print("Pred:  ",pred_mask_path)

                if gt_mask is not None and pred_mask is not None:
                    # Resize prediction mask to match GT mask shape if they are different
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = resize_mask(pred_mask, gt_mask.shape)

                    iou = calculate_iou(gt_mask, pred_mask, base=base_mode)
                    biou = boundary_iou(gt_mask, pred_mask, base=base_mode)
                    # print("IoU: ",iou," BIoU:   ",biou)
                    if cat_id not in iou_scores:
                        iou_scores[cat_id] = []
                        biou_scores[cat_id] = []
                    iou_scores[cat_id].append(iou)
                    biou_scores[cat_id].append(biou)
                    all_ious.append(iou)
                    all_bious.append(biou)

                    # Add metrics to per_view dictionary
                    per_view_dict[image_name][cat_id] = {
                        "IoU": iou,
                        "BIoU": biou
                    }

    # Accuracy computations
    total_count = len(all_ious)
    if total_count > 0:
        count_iou_025 = (np.array(all_ious) > 0.25).sum()
        count_iou_05 = (np.array(all_ious) > 0.5).sum()
        acc_iou_025 = float(count_iou_025 / total_count)
        acc_iou_05 = float(count_iou_05 / total_count)
        
        count_biou_025 = (np.array(all_bious) > 0.25).sum()
        count_biou_05 = (np.array(all_bious) > 0.5).sum()
        acc_biou_025 = float(count_biou_025 / total_count)
        acc_biou_05 = float(count_biou_05 / total_count)
    else:
        acc_iou_025 = acc_iou_05 = acc_biou_025 = acc_biou_05 = 0.0

    # Calculate mean IoU for each class
    mean_iou_per_class = {cat_id: float(np.mean(iou_scores[cat_id])) for cat_id in iou_scores}
    mean_biou_per_class = {cat_id: float(np.mean(biou_scores[cat_id])) for cat_id in biou_scores}

    # Calculate overall mean IoU
    overall_mean_iou = float(np.mean(all_ious)) if total_count > 0 else 0.0
    overall_mean_biou = float(np.mean(all_bious)) if total_count > 0 else 0.0

    print("\n--- Final Metrics ---")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")
    print(f"Overall Mean BIoU: {overall_mean_biou:.4f}")
    print(f"IoU Acc@0.25: {acc_iou_025:.4f} | Acc@0.5: {acc_iou_05:.4f}")
    print(f"BIoU Acc@0.25: {acc_biou_025:.4f} | Acc@0.5: {acc_biou_05:.4f}")
    print("Mean IoU per class:", mean_iou_per_class)
    print("Mean BIoU per class:", mean_biou_per_class)

    # Populate the aggregate dictionary
    full_dict = {
        "overall_mIoU": overall_mean_iou,
        "overall_mBIoU": overall_mean_biou,
        "IoU_Acc@0.25": acc_iou_025,
        "BIoU_Acc@0.25": acc_biou_025,
        "IoU_Acc@0.5": acc_iou_05,
        "BIoU_Acc@0.5": acc_biou_05,
        "per_class_mIoU": mean_iou_per_class,
        "per_class_mBIoU": mean_biou_per_class
    }


    results_path = os.path.join(out_dir, "results_segmentation_new.json")
    with open(results_path, 'w') as fp:
        json.dump(full_dict, fp, indent=4)
    
    per_view_path = os.path.join(out_dir, "per_view_segmentation_new.json")
    with open(per_view_path, 'w') as fp:
        json.dump(per_view_dict, fp, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser("Compute Segmentation IoU")
    parser.add_argument("--out_dir", type=str, required=True, 
                        help="Specify the out_dir, e.g, output/ramen, output/figurines, output/teatime")
    parser.add_argument("--base_mode", type=str, choices=["former", "later", "union"], default="union", 
                        help="Denominator to use for calculation ('former'=GT, 'later'=Pred, 'union'=Union)")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--iteration", type=str, default="70000")
    args = parser.parse_args()

    evaluate(args.out_dir, args.base_mode, args.split, args.iteration)