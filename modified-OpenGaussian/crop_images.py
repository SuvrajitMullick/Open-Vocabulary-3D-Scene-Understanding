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

def crop_images(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        path = os.path.join(dataset.source_path, "crop_images")
        makedirs(path, exist_ok=True)

        for _, view in enumerate(tqdm(all_cameras, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background, iteration, rescale=False)
            img = render_pkg["render"]
            torchvision.utils.save_image(img, os.path.join(path, view.image_name + ".png"))

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Cropping Images: " + args.model_path)

    safe_state(args.quiet)
    crop_images(model.extract(args), args.iteration, pipeline.extract(args))
    
# python crop_images.py -m output/ramen --iteration 30000