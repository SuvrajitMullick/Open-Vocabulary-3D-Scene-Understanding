#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image

from ext.grounded_sam import grouned_sam_output, load_model, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, train_test_exp, separate_sh, groundingdino_model, sam_predictor, threshold=0.2):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    
    # Text prompt
    if 'figurines' in model_path:
        positives = ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant']
    elif 'ramen' in model_path:
        positives = ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                    'napkin', 'bowl', 'glass of water', 'chopsticks', 'wavy noodles']
    elif 'teatime' in model_path:
        positives = ['sheep', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
                    'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                    'bag of cookies']
    else:
        raise NotImplementedError   # You can provide your text prompt here

    print("Text prompts:    ", positives)
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Evaluation frames
        if 'figurines' in model_path:
            candidate_frames = ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]
        elif 'ramen' in model_path:
            candidate_frames = ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]
        elif 'teatime' in model_path:
            candidate_frames = ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
        else:
            raise NotImplementedError   # You can provide your text prompt here
        
        if  view.image_name not in candidate_frames:
            continue
        
        pred_obj_rgbs_path = os.path.join(pred_obj_path, view.image_name, 'rgbs')
        pred_obj_masks_path = os.path.join(pred_obj_path, view.image_name, 'masks')
        makedirs(pred_obj_rgbs_path, exist_ok=True)
        makedirs(pred_obj_masks_path, exist_ok=True)
        
        results = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj_class_map = torch.argmax(logits,dim=0)
        
        image = (rendering.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
        for TEXT_PROMPT in positives:
            # Use Grounded-SAM
            text_mask, _ = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
            selected_obj_ids = select_obj_ioa(pred_obj_class_map, text_mask)

            if len(selected_obj_ids) > 0:
                pred_obj_prob = torch.softmax(logits,dim=0)
                pred_obj_mask = pred_obj_prob[selected_obj_ids, :, :] > threshold
                pred_obj_mask = pred_obj_mask.any(dim=0)
                pred_obj_mask = (pred_obj_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            else:
                pred_obj_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            bool_mask = (pred_obj_mask > 0)[..., None]
            pred_obj_rgb = np.where(bool_mask, image, 0).astype(np.uint8)
            
            Image.fromarray(pred_obj_rgb).save(os.path.join(pred_obj_rgbs_path, TEXT_PROMPT + ".png"))
            Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_masks_path, TEXT_PROMPT + ".png"))
            
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"), weights_only=True))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # grounding-dino
        ckpt_filepath = "checkpoints/groundingdino_swinb_cogcoor.pth"
        config_filepath = "config/GroundingDINO_SwinB_cfg.py"
        groundingdino_model = load_model(ckpt_filepath, config_filepath)

        # sam-hq
        sam_checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        if not skip_train:
            render_set(dataset.model_path, "text2obj_train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, dataset.train_test_exp, separate_sh, groundingdino_model, sam_predictor)
        if not skip_test:
            render_set(dataset.model_path, "text2obj_test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, dataset.train_test_exp, separate_sh, groundingdino_model, sam_predictor)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)