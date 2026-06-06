# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C

from typing import NamedTuple


class RasterSettings(NamedTuple):
    color_mode: str
    object_mode: str
    n_samp_per_vox: int
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    cx: float
    cy: float
    w2c_matrix: torch.Tensor
    c2w_matrix: torch.Tensor
    bg_color: float = 0
    near: float = 0.02
    need_depth: bool = False
    need_normal: bool = False
    track_max_w: bool = False
    lambda_R_concen: float = 0
    lambda_ascending: float = 0
    lambda_dist: float = 0
    # Optional gt color for color concnetration loss in backward pass.
    gt_color: torch.Tensor = torch.empty(0)
    debug: bool = False


def rasterize_voxels(
        raster_settings: RasterSettings,
        octree_paths: torch.Tensor,
        vox_centers: torch.Tensor,
        vox_lengths: torch.Tensor,
        vox_fn,
    ):

    # Some input checking
    if not isinstance(raster_settings, RasterSettings):
        raise Exception("Expect RasterSettings as first argument.")
    if raster_settings.n_samp_per_vox > _C.MAX_N_SAMP or raster_settings.n_samp_per_vox < 1:
        raise Exception(f"n_samp_per_vox should be in range [1, {_C.MAX_N_SAMP}].")

    N = octree_paths.numel()
    device = octree_paths.device
    if vox_centers.shape[0] != N or vox_lengths.numel() != N:
        raise Exception("Size mismatched.")
    if len(vox_centers.shape) != 2 or vox_centers.shape[1] != 3:
        raise Exception("Expect vox_centers in shape [N, 3].")
    if raster_settings.w2c_matrix.device != device or \
            raster_settings.c2w_matrix.device != device or \
            vox_centers.device != device or \
            vox_lengths.device != device:
        raise Exception("Device mismatch.")

    # Preprocess octree
    n_duplicates, geomBuffer = _C.rasterize_preprocess(
        raster_settings.image_width,
        raster_settings.image_height,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.cx,
        raster_settings.cy,
        raster_settings.w2c_matrix,
        raster_settings.c2w_matrix,
        raster_settings.near,

        octree_paths,
        vox_centers,
        vox_lengths,

        raster_settings.debug,
    )
    in_frusts_idx = torch.where(n_duplicates > 0)[0]

    # Forward voxel parameters
    cam_pos = raster_settings.c2w_matrix[:3, 3]
    vox_params = vox_fn(in_frusts_idx, cam_pos, raster_settings.color_mode, raster_settings.object_mode)
    geos = vox_params['geos']
    rgbs = vox_params['rgbs']
    objs = vox_params['objs']
    subdiv_p = vox_params['subdiv_p']

    # Some voxel parameters checking
    if geos.shape != (N, 8):
        raise Exception(f"Expect geos in ({N}, 8) but got", geos.shape)
    if rgbs.shape[0] != N:
        raise Exception(f"Expect rgbs in ({N}, 3) but got", rgbs.shape)
    if objs.shape[0] != N:
        raise Exception(f"Expect objs in ({N}, 16) but got", objs.shape)
    if subdiv_p.shape[0] != N:
        raise Exception(f"Expect subdiv_p in ({N}, 1) but got", subdiv_p.shape)

    if geos.device != device:
        raise Exception("Device mismatch: geos.")
    if rgbs.device != device:
        raise Exception("Device mismatch: rgbs.")
    if objs.device != device:
        raise Exception("Device mismatch: objs.")
    if subdiv_p.device != device:
        raise Exception("Device mismatch: subdiv_p.")

    # Some checking for regularizations
    if raster_settings.lambda_R_concen > 0:
        if len(raster_settings.gt_color.shape) != 3 or \
                raster_settings.gt_color.shape[0] != 3 or \
                raster_settings.gt_color.shape[1] != raster_settings.image_height or \
                raster_settings.gt_color.shape[2] != raster_settings.image_width:
            raise Exception("Except gt_color in shape of [3, H, W]")
        if raster_settings.gt_color.device != device:
            raise Exception("Device mismatch.")

    # Involk differentiable voxels rasterization.
    return _RasterizeVoxels.apply(
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        objs,
        subdiv_p,
    )


class _RasterizeVoxels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        objs,
        subdiv_p,
    ):

        need_distortion = raster_settings.lambda_dist > 0

        args = (
            raster_settings.n_samp_per_vox,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.bg_color,
            raster_settings.need_depth,
            need_distortion,
            raster_settings.need_normal,
            raster_settings.track_max_w,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            objs,

            geomBuffer,

            raster_settings.debug,
        )

        num_rendered, binningBuffer, imgBuffer, out_color, out_object, out_depth, out_normal, out_T, max_w = _C.rasterize_voxels(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            octree_paths, vox_centers, vox_lengths,
            geos, rgbs, objs,
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal)
        ctx.mark_non_differentiable(max_w)
        return out_color, out_object, out_depth, out_normal, out_T, max_w

    @staticmethod
    def backward(ctx, dL_dout_color, dL_dout_object, dL_dout_depth, dL_dout_normal, dL_dout_T, dL_dmax_w):
        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        octree_paths, vox_centers, vox_lengths, \
            geos, rgbs, objs, \
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal = ctx.saved_tensors

        args = (
            num_rendered,
            raster_settings.n_samp_per_vox,
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.bg_color,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            objs,

            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,

            dL_dout_color,
            dL_dout_object,
            dL_dout_depth,
            dL_dout_normal,
            dL_dout_T,

            raster_settings.lambda_R_concen,
            raster_settings.gt_color,
            raster_settings.lambda_ascending,
            raster_settings.lambda_dist,
            raster_settings.need_depth,
            raster_settings.need_normal,
            out_depth,
            out_normal,

            raster_settings.debug,
        )

        dL_dgeos, dL_drgbs, dL_dobjs, subdiv_p_bw = _C.rasterize_voxels_backward(*args)

        grads = (
            None, # => raster_settings
            None, # => geomBuffer
            None, # => octree_paths
            None, # => vox_centers
            None, # => vox_lengths
            dL_dgeos, # => geos
            dL_drgbs, # => rgbs
            dL_dobjs, # => objs
            subdiv_p_bw, # => subdivision priority
        )

        return grads


class SH_eval(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        active_sh_degree,
        idx,
        vox_centers, # Use dir to vox center
        cam_pos,
        viewdir,     # Use given dir
        sh0,
        shs,
    ):

        if torch.is_tensor(vox_centers) and vox_centers.requires_grad:
            raise NotImplementedError
        if torch.is_tensor(cam_pos) and cam_pos.requires_grad:
            raise NotImplementedError
        if torch.is_tensor(viewdir) and viewdir.requires_grad:
            raise NotImplementedError

        if idx is None:
            idx = torch.empty(0, dtype=torch.int64)

        if viewdir is not None:
            vox_centers = viewdir
            cam_pos = torch.zeros_like(cam_pos)

        rgbs = _C.sh_compute(
            active_sh_degree,
            idx,
            vox_centers,
            cam_pos,
            sh0,
            shs,
        )

        ctx.active_sh_degree = active_sh_degree
        ctx.M = 1 + shs.shape[1]
        ctx.save_for_backward(idx, vox_centers, cam_pos, rgbs)
        return rgbs

    @staticmethod
    def backward(ctx, dL_drgbs):
        # Restore necessary values from context
        idx, vox_centers, cam_pos, rgbs = ctx.saved_tensors
        dL_dsh0, dL_dshs = _C.sh_compute_bw(
            ctx.active_sh_degree,
            ctx.M,
            idx,
            vox_centers,
            cam_pos,
            rgbs,
            dL_drgbs,
        )

        grads = (
            None, # => active_sh_degree
            None, # => idx
            None, # => vox_centers
            None, # => cam_pos
            None, # => viewdir
            dL_dsh0,
            dL_dshs,
        )

        return grads


class SH_OBJ_eval(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        idx,
        vox_centers, # Use dir to vox center
        viewdir,     # Use given dir
        sh_objs,
    ):

        if torch.is_tensor(vox_centers) and vox_centers.requires_grad:
            raise NotImplementedError
        if torch.is_tensor(viewdir) and viewdir.requires_grad:
            raise NotImplementedError

        if idx is None:
            idx = torch.empty(0, dtype=torch.int64)

        if viewdir is not None:
            vox_centers = viewdir

        objs = _C.sh_obj_compute(
            idx,
            vox_centers,
            sh_objs,
        )

        ctx.save_for_backward(idx, vox_centers, objs)
        return objs

    @staticmethod
    def backward(ctx, dL_dobjs):
        # Restore necessary values from context
        idx, vox_centers, objs = ctx.saved_tensors
        dL_dsh_objs = _C.sh_obj_compute_bw(
            idx,
            vox_centers,
            objs,
            dL_dobjs,
        )

        grads = (
            None, # => idx
            None, # => vox_centers
            None, # => viewdir
            dL_dsh_objs,
        )

        return grads


class GatherGeoParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vox_key,
        care_idx,
        grid_pts,
    ):
        assert len(vox_key.shape) == 2 and vox_key.shape[1] == 8
        assert len(care_idx.shape) == 1
        assert grid_pts.shape[0] == grid_pts.numel()

        geo_params = _C.gather_triinterp_geo_params(vox_key, care_idx, grid_pts)

        ctx.num_grid_pts = grid_pts.numel()
        ctx.save_for_backward(vox_key, care_idx)
        return geo_params

    @staticmethod
    def backward(ctx, dL_dgeo_params):
        # Restore necessary values from context
        num_grid_pts = ctx.num_grid_pts
        vox_key, care_idx = ctx.saved_tensors

        dL_dgrid_pts = _C.gather_triinterp_geo_params_bw(vox_key, care_idx, num_grid_pts, dL_dgeo_params)

        return None, None, dL_dgrid_pts


class GatherFeatParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vox_key,
        care_idx,
        grid_pts,
    ):
        assert len(vox_key.shape) == 2 and vox_key.shape[1] == 8
        assert len(care_idx.shape) == 1
        assert len(grid_pts.shape) == 2

        feat_params = _C.gather_triinterp_feat_params(vox_key, care_idx, grid_pts)

        ctx.num_grid_pts = len(grid_pts)
        ctx.save_for_backward(vox_key, care_idx)
        return feat_params

    @staticmethod
    def backward(ctx, dL_dfeat_params):
        # Restore necessary values from context
        num_grid_pts = ctx.num_grid_pts
        vox_key, care_idx = ctx.saved_tensors

        dL_dgrid_pts = _C.gather_triinterp_feat_params_bw(vox_key, care_idx, num_grid_pts, dL_dfeat_params)

        return None, None, dL_dgrid_pts

def rasterize_voxels_apply_weights(
        raster_settings: RasterSettings,
        octree_paths: torch.Tensor,
        vox_centers: torch.Tensor,
        vox_lengths: torch.Tensor,
        geos: torch.Tensor,           # (P, 8)
        image_weights: torch.Tensor,  # (C, H, W)
        weights: torch.Tensor,        # (P, C)
        cnt: torch.Tensor,            # (P, C)
        geomBuffer: torch.Tensor,
    ):

    if not isinstance(raster_settings, RasterSettings):
        raise Exception("Expect RasterSettings as first argument.")

    H = raster_settings.image_height
    W = raster_settings.image_width

    if image_weights.ndimension() != 3:
        raise Exception("image_weights must have shape (C, H, W)")
    C, H_im, W_im = image_weights.shape
    if H_im != H or W_im != W:
        raise Exception(f"image_weights spatial size ({H_im}, {W_im}) -> must match ({H}, {W})")

    N = octree_paths.numel()
    if vox_centers.shape != (N, 3):
        raise Exception(f"vox_centers must be ({N}, 3), got {vox_centers.shape}")
    if vox_lengths.numel() != N:
        raise Exception(f"vox_lengths must have {N} elements, got {vox_lengths.numel()}")
    if geos.shape != (N, 8):
        raise Exception(f"geos must be ({N}, 8), got {geos.shape}")

    device = octree_paths.device
    for t, name in [
        (vox_centers, "vox_centers"),
        (vox_lengths, "vox_lengths"),
        (geos, "geos"),
        (image_weights, "image_weights"),
        (weights, "weights"),
        (cnt, "cnt"),
        (geomBuffer, "geomBuffer"),
        (raster_settings.w2c_matrix, "w2c_matrix"),
        (raster_settings.c2w_matrix, "c2w_matrix"),
    ]:
        if t.device != device:
            raise Exception(f"Device mismatch: {name}")
        
    args = (
        raster_settings.n_samp_per_vox,
        raster_settings.image_width,
        raster_settings.image_height,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.cx,
        raster_settings.cy,
        raster_settings.w2c_matrix,
        raster_settings.c2w_matrix,

        octree_paths,
        vox_centers,
        vox_lengths,
        geos,

        image_weights,
        weights,
        cnt,

        geomBuffer,
        raster_settings.debug,
    )

    _C.rasterize_voxels_apply_weights(*args)



def mark_n_duplicates(
        image_width, image_height,
        tanfovx, tanfovy,
        cx, cy,
        w2c_matrix, c2w_matrix, near,
        octree_paths, vox_centers, vox_lengths,
        return_buffer=False,
        debug=False):

    n_duplicates, geomBuffer = _C.rasterize_preprocess(
        image_width,
        image_height,
        tanfovx,
        tanfovy,
        cx,
        cy,
        w2c_matrix,
        c2w_matrix,
        near,

        octree_paths,
        vox_centers,
        vox_lengths,

        debug,
    )
    if return_buffer:
        return n_duplicates, geomBuffer
    return n_duplicates


def mark_max_samp_rate(cameras, octree_paths, vox_centers, vox_lengths, near=0.02):
    max_samp_rate = torch.zeros([len(octree_paths)], dtype=torch.float32, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        zdist = ((vox_centers - cam.position) * cam.lookat).sum(-1)
        vis_idx = torch.where((n_duplicates > 0) & (zdist > near))[0]
        zdist = zdist[vis_idx]
        samp_interval = zdist * cam.pix_size
        samp_rate = vox_lengths.squeeze(1)[vis_idx] / samp_interval
        max_samp_rate[vis_idx] = torch.maximum(max_samp_rate[vis_idx], samp_rate)
    return max_samp_rate


def mark_near(cameras, octree_paths, vox_centers, vox_lengths, near=0.2):
    is_near = torch.zeros([len(octree_paths)], dtype=torch.bool, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        vis_idx = torch.where(n_duplicates > 0)[0]
        zdist = ((vox_centers[vis_idx] - cam.position) * cam.lookat).sum(-1)
        is_near[vis_idx] |= (zdist <= near + vox_lengths.squeeze(1)[vis_idx])
    return is_near
