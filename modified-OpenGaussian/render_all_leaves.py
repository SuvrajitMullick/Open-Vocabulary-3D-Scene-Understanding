import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import numpy as np

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.opengs_utlis import load_code_book
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser


def render_all_leaves_visible_only(model_path, iteration, views, gaussians, pipeline, background):

    # -------------------------------
    # Load leaf indices
    # -------------------------------
    _, leaf_cluster_indices = load_code_book(
        os.path.join(model_path, "point_cloud", f'iteration_{iteration}', "leaf_code_book")
    )
    leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()

    # -------------------------------
    # Load mapping info
    # -------------------------------
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)

    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()

    # IMPORTANT: use final leaf mapping
    leaf_cluster_indices = leaf_ind

    total_leaf = leaf_cluster_indices.max().item() + 1
    print(f"Total leaf clusters: {total_leaf}")

    # -------------------------------
    # Output dir
    # -------------------------------
    base_output = os.path.join(model_path, "all_leaf_renders")
    makedirs(base_output, exist_ok=True)

    # -------------------------------
    # Loop over ALL views
    # -------------------------------
    for view in tqdm(views, desc="Processing all views"):

        if not view.data_on_gpu:
            view.to_gpu()

        # -------------------------------
        # Step 1: Find visible leaves
        # -------------------------------
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            iteration,
            rescale=False,
            leaf_cluster_idx=leaf_cluster_indices,
            render_feat_map=False,
            better_vis=False
        )

        occured_leaf_id = render_pkg["occured_leaf_id"]

        if len(occured_leaf_id) == 0:
            if view.data_on_gpu:
                view.to_cpu()
            continue

        occured_leaf_id = torch.tensor(occured_leaf_id).cuda()

        # -------------------------------
        # Step 2: Render each visible leaf
        # -------------------------------
        for leaf_id in occured_leaf_id:

            leaf_id_int = int(leaf_id.item())

            leaf_folder = os.path.join(base_output, f"leaf_{leaf_id_int:04d}")
            rgb_folder = os.path.join(leaf_folder, "rgb")
            mask_folder = os.path.join(leaf_folder, "mask")

            makedirs(rgb_folder, exist_ok=True)
            makedirs(mask_folder, exist_ok=True)

            selected_leaf = leaf_id.unsqueeze(0)

            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                iteration,
                rescale=False,
                leaf_cluster_idx=leaf_cluster_indices,
                selected_leaf_id=selected_leaf,
                render_feat_map=False,
                render_cluster=False,
                better_vis=True,
                seg_rgb=True,
                post_process=True
            )

            rendered_imgs = render_pkg["leaf_clusters_imgs"]
            silhouettes = render_pkg["leaf_cluster_silhouettes"]

            if len(rendered_imgs) == 0:
                continue

            for i in range(len(rendered_imgs)):

                # RGB
                rgb = rendered_imgs[i][:3, :, :]
                rgb_path = os.path.join(rgb_folder, f"{view.image_name}.png")
                torchvision.utils.save_image(rgb, rgb_path)

                # MASK
                mask = (silhouettes[i] > 0.7).float()
                mask_path = os.path.join(mask_folder, f"{view.image_name}.png")
                torchvision.utils.save_image(mask, mask_path)

        if view.data_on_gpu:
            view.to_cpu()


def render_sets(dataset, iteration, pipeline):

    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        print("Rendering ALL views (visible leaves only)...")
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()

        render_all_leaves_visible_only(
            dataset.model_path,
            scene.loaded_iter,
            all_cameras,
            gaussians,
            pipeline,
            background
        )


if __name__ == "__main__":

    parser = ArgumentParser(description="Render visible leaf clusters in ALL views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    print("Model path:", args.model_path)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args)
    )