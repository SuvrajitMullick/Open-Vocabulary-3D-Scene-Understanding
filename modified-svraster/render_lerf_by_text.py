# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Adapted from render_svraster.py with LERF/Grounded-SAM text-prompted
# segmentation, following the approach in render_lerf_mask_gaussian_grouping.py.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio
from PIL import Image

import torch

from src.config import cfg, update_argparser, update_config
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np

from ext.grounded_sam import grouned_sam_output, load_model, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor


@torch.no_grad()
def render_set(name, iteration, args, views, voxel_model, classifier, groundingdino_model, sam_predictor, threshold=0.2):
    im_render_path = os.path.join(args.model_path, name, f"ours_{iteration}", "renders")
    im_gts_path = os.path.join(args.model_path, name, f"ours_{iteration}", "gt")
    obj_render_path = os.path.join(args.model_path, name, f"ours_{iteration}", "objects_render")
    makedirs(im_render_path, exist_ok=True)
    makedirs(im_gts_path, exist_ok=True)
    makedirs(obj_render_path, exist_ok=True)
    print(f'im_render_path: {im_render_path}')
    print(f'ss            =: {voxel_model.ss}')
    print(f'n_samp_per_vox=: {voxel_model.n_samp_per_vox}')
    
    # Text prompt
    if 'figurines' in args.model_path:
        positives = ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant']
    elif 'ramen' in args.model_path:
        positives = ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                    'napkin', 'bowl', 'glass of water', 'chopsticks', 'wavy noodles']
    elif 'teatime' in args.model_path:
        positives = ['sheep', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
                    'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                    'bag of cookies']
    else:
        raise NotImplementedError   # You can provide your text prompt here

    print("Text prompts:    ", positives)

    for _, view in enumerate(tqdm(views, desc=f"Rendering progress")):
        # Evaluation frames
        if 'figurines' in args.model_path:
            candidate_frames = ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]
        elif 'ramen' in args.model_path:
            candidate_frames = ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]
        elif 'teatime' in args.model_path:
            candidate_frames = ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
        else:
            raise NotImplementedError   # You can provide your text prompt here
        
        if  view.image_name not in candidate_frames:
            continue
        obj_render_rgbs_path = os.path.join(obj_render_path, view.image_name, 'rgbs')
        obj_render_masks_path = os.path.join(obj_render_path, view.image_name, 'masks')
        makedirs(obj_render_rgbs_path, exist_ok=True)
        makedirs(obj_render_masks_path, exist_ok=True)
        
        render_pkg = voxel_model.render(view)
        rendering = render_pkg['color']
        rendering_obj = render_pkg['object']
        logits = classifier(rendering_obj)
        obj_render_class_map = torch.argmax(logits, dim=0)
        
        image = (rendering.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
        for TEXT_PROMPT in positives:
            # Use Grounded-SAM
            text_mask, _ = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
            selected_obj_ids = select_obj_ioa(obj_render_class_map, text_mask)

            if len(selected_obj_ids) > 0:
                object_render_prob = torch.softmax(logits, dim=0)
                object_render_mask = object_render_prob[selected_obj_ids, :, :] > threshold
                object_render_mask = object_render_mask.any(dim=0)
                object_render_mask = (object_render_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            else:
                object_render_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
            bool_mask = (object_render_mask > 0)[..., None]
            object_render_rgb = np.where(bool_mask, image, 0).astype(np.uint8)

            
            Image.fromarray(object_render_rgb).save(os.path.join(obj_render_rgbs_path, TEXT_PROMPT + ".png"))
            Image.fromarray(object_render_mask).save(os.path.join(obj_render_masks_path, TEXT_PROMPT + ".png"))

        im_gt = view.image.cuda()
        ext = ".jpg" if args.use_jpg else ".png"
        imageio.imwrite(os.path.join(im_render_path, view.image_name + ext),  im_tensor2np(rendering))
        imageio.imwrite(os.path.join(im_gts_path, view.image_name + ".png"), im_tensor2np(im_gt))


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering text-prompted masks.")
    parser.add_argument('--model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--rgb_only", action="store_true")
    parser.add_argument("--use_jpg", action="store_true")
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_n_samp_per_vox", default=None)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        object_dir_name=cfg.data.object_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
    )

    # Load model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)

    # Output path suffix
    suffix = args.suffix
    if not args.suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"
    
    if args.overwrite_n_samp_per_vox:
        voxel_model.n_samp_per_vox = args.overwrite_n_samp_per_vox
        if not args.suffix:
            suffix += f"_{args.overwrite_n_samp_per_vox}"

    voxel_model.freeze_vox_geo()
    
    num_classes = cfg.data.num_classes
    print("Num classes: ",num_classes)

    classifier = torch.nn.Conv2d(voxel_model.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(args.model_path,"checkpoints",'classifier.pth'), weights_only=True))

    # grounding-dino
    ckpt_filepath    = "checkpoints/groundingdino_swinb_cogcoor.pth"
    config_filepath  = "cfg/GroundingDINO_SwinB_cfg.py"
    groundingdino_model = load_model(ckpt_filepath, config_filepath)

    # sam-hq
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)

    if not args.skip_train:
        render_set(
            "text2obj_train", loaded_iter, args,
            data_pack.get_train_cameras(), voxel_model, classifier,
            groundingdino_model, sam_predictor)
        
    if not args.skip_test:
        render_set(
            "text2obj_test", loaded_iter, args, 
            data_pack.get_test_cameras(), voxel_model, classifier,
            groundingdino_model, sam_predictor)