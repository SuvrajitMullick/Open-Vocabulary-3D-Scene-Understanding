#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, OptimizationParams, get_combined_args

def crop_scene(dataset : ModelParams, opt : OptimizationParams, iteration : int, padding : float):

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.training_setup(opt)

    chckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
    checkpoint = torch.load(chckpt_path, map_location='cuda')
    model_params, _ = checkpoint

    if len(model_params) == 12:
        ins_feat = torch.rand((model_params[8].shape[0], opt.ins_feat_dim), dtype=torch.float, device="cuda")
        ins_feat = torch.nn.Parameter(ins_feat.requires_grad_(True))
        to_list = list(model_params)
        to_list[10] = gaussians.optimizer.state_dict()
        to_list.insert(7, ins_feat)
        ins_feat_q = torch.empty(0)
        to_list.insert(8, ins_feat_q)
        model_params = tuple(to_list)
    gaussians.restore(model_params, opt)

    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    cam_positions = [cam.camera_center.detach().cpu().numpy() for cam in all_cameras]
    cam_positions = np.array(cam_positions)
    centroid = np.mean(cam_positions, axis=0)

    distances = np.linalg.norm(cam_positions - centroid, axis=1)
    radius = np.max(distances) * padding
    points = gaussians.get_xyz.detach().cpu().numpy()
    point_distances = np.linalg.norm(points - centroid, axis=1)
    prune_mask = point_distances > radius
    prune_mask_tensor = torch.tensor(prune_mask, device="cuda", dtype=torch.bool)

    orig_count = gaussians.get_xyz.shape[0]
    crop_count = (~prune_mask).sum()
    print(f"-> Original Gaussians: {orig_count}")
    print(f"-> Gaussians remaining: {crop_count} (Removed {orig_count - crop_count})")

    gaussians.prune_points(prune_mask_tensor)
    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    scene.save(f"{iteration}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Cropper")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--padding", default=1.5, type=float, help="Multiplier for the bounding sphere radius")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Cropping Target Model: " + args.model_path)

    safe_state(args.quiet)
    crop_scene(lp.extract(args), op.extract(args), args.iteration, args.padding)
    
# python crop_scene.py -m output/ramen --iteration 30000
