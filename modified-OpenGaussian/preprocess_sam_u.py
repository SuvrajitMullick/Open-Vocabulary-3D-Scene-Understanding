import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sam_langsplat import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def set_negatives(self, text_list):
        self.negatives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = self.model.encode_text(tok_phrases)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


class ContextAwareSemanticHierarchicalUnifier:
    def __init__(self, clip_model):
        self.clip = clip_model
        
        # Base robust text prompts
        # self.clip.set_positives([
        #     "a recognizable piece of food, pastry, or meal ingredient",
        #     "a ceramic bowl, teacup, teapot, or plate",
        #     "a miniature anime figurine, action figure, or plastic toy",
        #     "a dining utensil, spoon, or pair of chopsticks",
        #     "a distinct small household item or accessory",
        #     "a standalone 3D object resting on a surface"
        # ])
        # self.clip.set_negatives([
        #     "a flat wooden table, tablecloth, or tabletop surface", 
        #     "a blurry, out-of-focus background or depth of field effect",
        #     "a flat room wall or background architecture",
        #     "a dark cast shadow on a surface",
        #     "a bright glare, reflection, or lighting artifact",
        #     "empty background space or uniform flat texture",
        #     "unstructured background clutter or floating dust"
        # ])
        self.clip.set_positives([
            "a single, distinct, standalone object",
            "one individual item completely separated from others",
            "a single cohesive structure without distinct sub-parts",
            "a single individual piece of food or single utensil",
            "one solid individual toy or figure"
        ])
        self.clip.set_negatives([
            "a group, collection, or cluster of multiple distinct objects",
            "several different items physically touching or grouped together",
            "a complex scene with many separate interacting parts",
            "a bowl, plate, or scene containing multiple different pieces of food",
            "a flat background surface, wooden table, or tabletop",
            "a blurry, out-of-focus background region"
        ])
            
    def _lerp(self, score, min_val, max_val):
        """Bounded Linear Interpolation"""
        return min_val + score * (max_val - min_val)

    def _extract_features_and_complexity(self, mask, image):
        """Extracts Semantics, Appearance, and Regional Complexity (Entropy)."""
        x, y, w, h = np.int32(mask['bbox'])
        
        # 1. RGB Crop for CLIP
        img_copy = image.copy()
        img_copy[mask['segmentation'] == 0] = 0
        cropped = img_copy[y:y+h, x:x+w, ...]
        l = max(w, h)
        pad = np.zeros((l, l, 3), dtype=np.uint8)
        if h > w:
            pad[:, (h-w)//2:(h-w)//2 + w, :] = cropped
        else:
            pad[(w-h)//2:(w-h)//2 + h, :, :] = cropped
        pad_resized = cv2.resize(pad, (224, 224))
        
        crop_tensor = torch.from_numpy(pad_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embed = self.clip.encode_image(crop_tensor.to("cuda"))
            embed /= embed.norm(dim=-1, keepdim=True)
            
            sims_pos = torch.mm(embed, self.clip.pos_embeds.T).max(dim=-1)[0]
            sims_neg = torch.mm(embed, self.clip.neg_embeds.T).max(dim=-1)[0]
            
            temperature = 25.0 
            margin = sims_pos - sims_neg
            score = torch.sigmoid(temperature * margin)
            
        mask['objectness'] = score.item()
        mask['embed'] = embed.cpu()

        # 2. HS Histogram & Shannon Entropy (Complexity)
        raw_crop = image[y:y+h, x:x+w, ...]
        crop_hsv = cv2.cvtColor(raw_crop, cv2.COLOR_RGB2HSV)
        crop_bool = mask['segmentation'][y:y+h, x:x+w].astype(np.uint8)
        
        hist = cv2.calcHist([crop_hsv], [0, 1], crop_bool, [16, 16], [0, 180, 0, 256])
        
        # Calculate Shannon Entropy of the region to determine its complexity
        hist_pmf = hist / (hist.sum() + 1e-7)
        non_zero = hist_pmf[hist_pmf > 0]
        entropy = -np.sum(non_zero * np.log2(non_zero))
        
        # Max entropy for 256 bins is 8. Bounding realistic max to 6.0 for better sensitivity.
        mask['complexity'] = min(entropy / 6.0, 1.0) 
        
        # Normalize histogram for correlation comparisons
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        mask['hist'] = hist.flatten()
        return mask

    def build_containment_matrix(self, parent_masks, child_masks):
        if not parent_masks or not child_masks: 
            return torch.zeros((len(parent_masks), len(child_masks)))
        
        P = torch.from_numpy(np.stack([m['segmentation'] for m in parent_masks])).float()
        C = torch.from_numpy(np.stack([m['segmentation'] for m in child_masks])).float()
        
        intersection = torch.einsum('nhw,mhw->nm', P, C)
        child_areas = C.sum(dim=(1, 2)).unsqueeze(0)
        containment = intersection / (child_areas + 1e-6)
        return containment

    def unify(self, masks_l, masks_m, masks_s, image):
        # Process all levels to extract features
        all_masks = list(masks_l) + list(masks_m) + list(masks_s)
        for m in all_masks:
            self._extract_features_and_complexity(m, image)

        final_masks = []
        
        # Build containment matrices for Large -> Medium ONLY
        contain_m_in_l = self.build_containment_matrix(masks_l, masks_m)

        claimed_m = set()

        # --- STEP 1: Top-Down Hierarchy (Cascading Texture Evaluation) ---
        for i, parent_l in enumerate(masks_l):
            children_m_idx = torch.where(contain_m_in_l[i] > 0.8)[0].tolist()
            claimed_m.update(children_m_idx)
            num_children_m = len(children_m_idx)

            # L -> M Variation Check
            if (num_children_m <= 1 or 
                parent_l['complexity'] <= 0.45 or parent_l['complexity'] >= 0.85 or 
                parent_l['objectness'] <= 0.45 or parent_l['objectness'] >= 0.85):
                final_masks.append(parent_l)
                continue
                
            # L -> M Diversity Check
            total_sim_m = 0.0
            weight_sum_m = 0.0
            for c1 in range(num_children_m):
                child_a = masks_m[children_m_idx[c1]]
                area_a = child_a['segmentation'].sum()
                for c2 in range(c1 + 1, num_children_m):
                    child_b = masks_m[children_m_idx[c2]]
                    area_b = child_b['segmentation'].sum()
                    sim = cv2.compareHist(child_a['hist'], child_b['hist'], cv2.HISTCMP_CORREL)
                    weight = area_a * area_b
                    total_sim_m += sim * weight
                    weight_sum_m += weight
            avg_visual_sim_m = total_sim_m / (weight_sum_m + 1e-6)
            
            if avg_visual_sim_m <= 0.25 or avg_visual_sim_m >= 0.75:
                final_masks.append(parent_l)
                continue
            
            # L -> M Threshold
            obj_score = parent_l['objectness']
            nonlinear_factor = 1.0 - (obj_score ** 2)
            dyn_split_thresh_m = self._lerp(nonlinear_factor, min_val=0.4, max_val=0.85)
            
            if avg_visual_sim_m <= dyn_split_thresh_m:
                for m_idx in children_m_idx:
                    final_masks.append(masks_m[m_idx])
            else:
                final_masks.append(parent_l)

        # --- STEP 2: Orphan Recovery (Vacuum-Filling Only) ---
        H, W = image.shape[:2]
        foreground_canvas = np.zeros((H, W), dtype=bool)
        
        for m in final_masks:
            if m['objectness'] > 0.1: 
                foreground_canvas = np.logical_or(foreground_canvas, m['segmentation'])

        # Evaluate Medium Orphans to fill the vacuum
        for m_idx, m_mask in enumerate(masks_m):
            if m_idx not in claimed_m:
                orphan_area = m_mask['segmentation'].sum()
                overlap_area = np.logical_and(m_mask['segmentation'], foreground_canvas).sum()
                overlap_ratio = overlap_area / (orphan_area + 1e-6)
                
                if overlap_ratio < 0.50:
                    final_masks.append(m_mask)
                    foreground_canvas = np.logical_or(foreground_canvas, m_mask['segmentation'])
                    
        # Evaluate Small Orphans to fill any remaining micro-vacuums
        for s_mask in masks_s:
            orphan_area = s_mask['segmentation'].sum()
            overlap_area = np.logical_and(s_mask['segmentation'], foreground_canvas).sum()
            overlap_ratio = overlap_area / (orphan_area + 1e-6)
            
            if overlap_ratio < 0.50:
                final_masks.append(s_mask)
                foreground_canvas = np.logical_or(foreground_canvas, s_mask['segmentation'])
        
        # --- STEP 3: Resolve Overlaps ---
        return self._resolve_overlaps(final_masks, image.shape[:2])

    def _resolve_overlaps(self, masks, shape):
        """Ensures the final mask layer is strictly non-overlapping"""
        # Sort lowest objectness to highest (Highest objectness gets placed last / on top)
        masks.sort(key=lambda x: x['objectness'])
        canvas = -np.ones(shape, dtype=np.int32)
        
        for i, m in enumerate(masks):
            canvas[m['segmentation']] = i
            
        resolved_masks = []
        for i, m in enumerate(masks):
            final_seg = (canvas == i)
            if final_seg.sum() > 50: # Prune micro-slivers left after overwrites
                m['segmentation'] = final_seg
                m['bbox'] = cv2.boundingRect(final_seg.astype(np.uint8))
                resolved_masks.append(m)
                
        return resolved_masks


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    # INITIALIZE AS 1 CHANNEL
    seg_maps = torch.zeros((len(image_list), 1, *image_list[0].shape[1:]), dtype=torch.int32) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder_fn):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder_fn(aug_imgs)

    clip_embeds = {}
    for mode in seg_images.keys():
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0 and scores.numel() > 0:
        index = scores.topk(min(3, scores.numel())).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0 and scores.numel() > 0:
        index = scores.topk(min(3, scores.numel())).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0 and scores.numel() > 0:
        index = scores.topk(min(3, scores.numel())).indices
        keep_inner_l[index] = True
        
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if len(masks_lvl) == 0:
            masks_new += ([],)
            continue
        
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    masks_unified = unifier.unify(masks_l, masks_m, masks_s, image)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['unified'], seg_maps['unified'] = mask2segmap(masks_unified, image)

    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--crop', action='store_true', help='Enable cropping')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    image_dir = "crop_images" if args.crop else "images"
    img_folder = os.path.join(dataset_path, image_dir)
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    unifier = ContextAwareSemanticHierarchicalUnifier(clip_model=model)
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    prefix = "crop_" if args.crop else ""
    save_folder = os.path.join(dataset_path, prefix + 'language_features_sam_u')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)