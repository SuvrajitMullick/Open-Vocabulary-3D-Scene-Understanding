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


def render_all_leaf_clusters_together(model_path, iteration, views, gaussians, pipeline, background):

    # -------------------------------
    # 1. Determine root_num and leaf_num
    # -------------------------------
    _, root_cluster_indices = load_code_book(
        os.path.join(model_path, "point_cloud", f'iteration_{iteration}', "root_code_book")
    )
    root_num = root_cluster_indices.max().item() + 1

    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)

    leaf_cluster_indices = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()
    leaf_lang_feat = saved_data["leaf_feat.npy"]
    
    total_leaves = leaf_lang_feat.shape[0]
    leaf_num = total_leaves // root_num

    print(f"Total Roots: {root_num}, Leaves per root: {leaf_num}")

    # Reconstruct coarse cluster indices for the visibility check
    cluster_indices = leaf_cluster_indices // leaf_num

    # -------------------------------
    # 2. Setup Output Directories
    # -------------------------------
    base_output = os.path.join(model_path, "grouped_leaf_renders")
    makedirs(base_output, exist_ok=True)

    # -------------------------------
    # 3. Loop over ALL views
    # -------------------------------
    for view in tqdm(views, desc="Processing all views"):

        if not view.data_on_gpu:
            view.to_gpu()

        # -------------------------------
        # Step A: Find visible coarse clusters
        # -------------------------------
        render_pkg_coarse = render(
            view,
            gaussians,
            pipeline,
            background,
            iteration,
            rescale=False,
            cluster_idx=cluster_indices,
            render_feat_map=False,
            render_cluster=True,   # Required to populate cluster_occur
            better_vis=True,
            selected_root_id=None
        )

        cluster_occur = render_pkg_coarse["cluster_occur"]

        if cluster_occur is None or not cluster_occur.any():
            if view.data_on_gpu:
                view.to_cpu()
            continue

        occured_cluster_id = torch.where(cluster_occur)[0]

        # -------------------------------
        # Step B: Render the leaves of each visible cluster TOGETHER
        # -------------------------------
        for cluster_id in occured_cluster_id:

            cluster_id_int = int(cluster_id.item())

            cluster_folder = os.path.join(base_output, f"cluster_{cluster_id_int:04d}")
            rgb_folder = os.path.join(cluster_folder, "rgb")
            mask_folder = os.path.join(cluster_folder, "mask")

            makedirs(rgb_folder, exist_ok=True)
            makedirs(mask_folder, exist_ok=True)

            # Group all leaf IDs belonging to this root cluster into a 1D tensor
            start_leaf = cluster_id_int * leaf_num
            end_leaf = start_leaf + leaf_num
            target_leaves = torch.arange(start_leaf, end_leaf).cuda()

            # Render the leaves
            render_pkg_leaf = render(
                view,
                gaussians,
                pipeline,
                background,
                iteration,
                rescale=False,
                cluster_idx=cluster_indices,           
                leaf_cluster_idx=leaf_cluster_indices, 
                selected_root_id=cluster_id_int,       
                selected_leaf_id=target_leaves,        # Passing the tensor merges them in your render function
                render_feat_map=False,
                render_cluster=False,                  # Skip coarse render here
                better_vis=True,
                seg_rgb=True,
                post_process=True,
                root_num=root_num,
                leaf_num=leaf_num
            )

            rendered_imgs = render_pkg_leaf["leaf_clusters_imgs"]
            silhouettes = render_pkg_leaf["leaf_cluster_silhouettes"]

            if rendered_imgs is None or len(rendered_imgs) == 0:
                continue

            # Because `selected_leaf_id` was a tensor, the render loop matches all of them 
            # simultaneously and appends a single grouped image/mask to index 0.
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

        print("Rendering ALL grouped leaf clusters (visible only)...")
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()

        render_all_leaf_clusters_together(
            dataset.model_path,
            scene.loaded_iter,
            all_cameras,
            gaussians,
            pipeline,
            background
        )


if __name__ == "__main__":

    parser = ArgumentParser(description="Render combined leaf clusters in ALL views")
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