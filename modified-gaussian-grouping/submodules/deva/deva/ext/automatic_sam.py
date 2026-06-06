# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything

from typing import Dict, List, Optional, Tuple, Type
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from segment_anything import sam_model_registry, sam_model_registry_baseline
from deva.ext.SAM.automatic_mask_generator import SamAutomaticMaskGenerator
from deva.ext.SAM.automatic_unified_mask_generator import UnifiedSamAutomaticMaskGenerator
# from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from deva.inference.object_info import ObjectInfo

import cv2
from dataclasses import dataclass, field

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


def get_sam_model(config: Dict, device: str):
    variant = config['sam_variant'].lower()
    # if variant == 'mobile':
    #     MOBILE_SAM_CHECKPOINT_PATH = config['MOBILE_SAM_CHECKPOINT_PATH']

        # Building Mobile SAM model
        # checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH, weights_only=True)
        # mobile_sam = setup_mobile_sam()
        # mobile_sam.load_state_dict(checkpoint, strict=True)
        # mobile_sam.to(device=device)
        # auto_sam = SamAutomaticMaskGenerator(mobile_sam,
        #                                      points_per_side=config['SAM_NUM_POINTS_PER_SIDE'],
        #                                      points_per_batch=config['SAM_NUM_POINTS_PER_BATCH'],
        #                                      pred_iou_thresh=config['SAM_PRED_IOU_THRESHOLD'])
    SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
    if variant == 'sam':
        SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry_baseline[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
            device=device)
        auto_sam = SamAutomaticMaskGenerator(sam,
                                             points_per_side=config['SAM_NUM_POINTS_PER_SIDE'],
                                             points_per_batch=config['SAM_NUM_POINTS_PER_BATCH'],
                                             pred_iou_thresh=config['SAM_PRED_IOU_THRESHOLD'])
    elif variant == 'sam_hq':
        HQ_SAM_CHECKPOINT_PATH = config['HQ_SAM_CHECKPOINT_PATH']

        # Building HQ_SAM Model and HQ_SAM Predictor
        hq_sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=HQ_SAM_CHECKPOINT_PATH).to(
            device=device)
        auto_sam = SamAutomaticMaskGenerator(hq_sam,
                                             points_per_side=config['SAM_NUM_POINTS_PER_SIDE'],
                                             points_per_batch=config['SAM_NUM_POINTS_PER_BATCH'],
                                             pred_iou_thresh=config['SAM_PRED_IOU_THRESHOLD'])    
    elif variant == 'sam_unified':
        SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']

        # Building UNIFIED_SAM Model and UNIFIED Predictor
        unified_sam = sam_model_registry_baseline[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
            device=device)
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        unifier = ContextAwareSemanticHierarchicalUnifier(clip_model=model)
        auto_sam = UnifiedSamAutomaticMaskGenerator(unified_sam,
                                             points_per_side=32,
                                             pred_iou_thresh=0.7,
                                             box_nms_thresh=0.7,
                                             stability_score_thresh=0.85,
                                             crop_n_layers=1,
                                             crop_n_points_downscale_factor=1,
                                             min_mask_region_area=100,           
                                             unifier = unifier,
                                             nms_iou_thr = 0.8,
                                             nms_score_thr = 0.7,
                                             nms_inner_thr = 0.5)
    else:
        raise ValueError(f'Unknown SAM variant: {config["SAM_VARIANT"]}')

    return auto_sam


def auto_segment(config: Dict, auto_sam, image: np.ndarray,
                 forward_mask: Optional[torch.Tensor], min_side: int,
                 suppress_small_mask: bool) -> Tuple[torch.Tensor, List[ObjectInfo]]:
    """
    config: the global configuration dictionary
    image: the image to segment; should be a numpy array; H*W*3; unnormalized (0~255)
    forward_mask: the mask used to determine positive/negative points; H*W

    Returns: a torch index mask of the same size as image; H*W
             a list of segment info, see object_utils.py for definition
    """
    device = auto_sam.predictor.device

    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    if forward_mask is not None:
        # compute positive and negative points
        foreground_mask = (forward_mask > 0).float().unsqueeze(0).unsqueeze(0)
        foreground_mask = F.interpolate(foreground_mask,
                                        scale_factor=1 / 16,
                                        mode='bilinear',
                                        antialias=True)  # blurring
        n_per_side = config['SAM_NUM_POINTS_PER_SIDE']
        offset = 1 / (2 * n_per_side)
        points_one_side = torch.linspace(offset, 1 - offset, n_per_side, device=device)
        points_x = points_one_side.unsqueeze(0).repeat(n_per_side, 1)
        points_y = points_one_side.unsqueeze(1).repeat(1, n_per_side)
        points = torch.stack([points_x, points_y], dim=-1).unsqueeze(0)
        points_label = F.grid_sample(foreground_mask, points * 2 - 1, align_corners=False).view(-1)
        points = points.view(-1, 2)
        positive_points = points[points_label < 0.01].cpu().numpy()
        if len(positive_points) == 0:
            output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
            segments_info = []
            return output_mask, segments_info
        # negative_points = points[points_label >= 0.5].cpu().numpy()
        negative_points = None  # no negative points
        mask_data = auto_sam.generate(image, positive_points, negative_points)
    else:
        mask_data = auto_sam.generate(image)

    curr_id = 1
    segments_info = []

    pred_masks = mask_data['masks'].float()  # num masks * H * W
    predicted_iou = mask_data["iou_preds"]

    # score mask by their areas
    if pred_masks.shape[0] == 0:
        output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
    else:
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), (new_h, new_w), mode='bilinear')[0]

        curr_id = 1
        if suppress_small_mask:
            areas = pred_masks.flatten(-2).sum(-1)
            scores = areas.unsqueeze(-1).unsqueeze(-1)

            scored_masks = pred_masks * scores
            scored_masks_with_bg = torch.cat(
                [torch.zeros((1, *pred_masks.shape[1:]), device=device) + 0.1, scored_masks], dim=0)
            output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)

            # let large mask eats small masks (small/tiny/incomplete masks are too common in SAM)
            hard_mask = torch.argmax(scored_masks_with_bg, dim=0)
            for k in range(scores.shape[0]):
                mask_area = (hard_mask == (k + 1)).sum()
                original_area = (pred_masks[k] > 0.5).sum()
                mask = (hard_mask == (k + 1)) & (pred_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum() > 0:
                    if mask_area / original_area < config['SAM_OVERLAP_THRESHOLD']:
                        continue
                    output_mask[mask] = curr_id
                    segments_info.append(ObjectInfo(id=curr_id, score=predicted_iou[k].item()))
                    curr_id += 1
        else:
            # prefer smaller objects
            areas = pred_masks.flatten(-2).sum(-1)
            scores = (areas.max() * 2 - areas).unsqueeze(-1).unsqueeze(-1)
            scored_masks = pred_masks * scores

            # add background channel
            scored_masks_with_bg = torch.cat(
                [torch.zeros((1, *scored_masks.shape[1:]), device=device) + 0.1, scored_masks],
                dim=0)
            output_mask = torch.argmax(scored_masks_with_bg, dim=0)
            for k in range(scored_masks.shape[0]):
                mask = (output_mask == (k + 1))
                if mask.sum() > 0:
                    segments_info.append(ObjectInfo(id=curr_id, score=predicted_iou[k].item()))
                    curr_id += 1

    return output_mask, segments_info
