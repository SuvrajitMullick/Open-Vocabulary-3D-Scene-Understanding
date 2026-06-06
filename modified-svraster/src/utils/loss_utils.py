# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

from lpips import LPIPS
from pytorch_msssim import SSIM
from fused_ssim import fused_ssim


metric_networks = {}


def l1_loss(x, y):
    return torch.nn.functional.l1_loss(x, y)

def l2_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)

def huber_loss(x, y, thres=0.01):
    l1 = (x - y).abs().mean(0)
    l2 = (x - y).pow(2).mean(0)
    loss = torch.where(
        l1 < thres,
        l2,
        2 * thres * l1 - thres ** 2)
    return loss.mean()

def cauchy_loss(x, y, reduction='mean'):
    loss_map = torch.log1p(torch.square(x - y))
    if reduction == 'sum':
        return loss_map.sum()
    if reduction == 'mean':
        return loss_map.mean()
    raise NotImplementedError

def psnr_score(x, y):
    return -10 * torch.log10(l2_loss(x, y))

def ssim_score(x, y):
    if 'SSIM' not in metric_networks:
        metric_networks['SSIM'] = SSIM(data_range=1, win_size=11, win_sigma=1.5, channel=3).cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks['SSIM'](x, y)

def ssim_loss(x, y):
    return 1 - ssim_score(x, y)

def fast_ssim_loss(x, y):
    # Note! Only x get gradient in backward.
    is_train = x.requires_grad or y.requires_grad
    return 1 - fused_ssim(x.unsqueeze(0), y.unsqueeze(0), padding="valid", train=is_train)

def lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x, y)

def correct_lpips_loss(x, y, net='vgg'):
    key = f'LPIPS_{net}'
    if key not in metric_networks:
        metric_networks[key] = LPIPS(net=net, version='0.1').cuda()
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(y.shape) == 3:
        y = y.unsqueeze(0)
    return metric_networks[key](x*2-1, y*2-1)

def entropy_loss(prob):
    pos_prob = prob.clamp(1e-6, 1-1e-6)
    neg_prob = 1 - pos_prob
    return -(pos_prob * pos_prob.log() + neg_prob * neg_prob.log()).mean()

def prob_concen_loss(prob):
    return (prob.square() * (1 - prob).square()).mean()


def exp_anneal(end_mul, iter_now, iter_from, iter_end):
    if end_mul == 1 or iter_now >= iter_end:
        return 1
    total_len = iter_end - iter_from + 1
    now_len = max(0, iter_now - iter_from + 1)
    now_p = min(1.0, now_len / total_len)
    return end_mul ** now_p


def loss_cls_3D(features, predictions, k=10, max_points=200000, sample_size=2500):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k+1, largest=False)  # Get top-k smallest distances
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]  # remove self index

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    return loss


def cosine_similarity(f1, f2):
    f1_norm = f1 / (f1.norm(dim=1, keepdim=True) + 1e-8)
    f2_norm = f2 / (f2.norm(dim=1, keepdim=True) + 1e-8)
    return (f1_norm * f2_norm).sum(dim=1) # [N]


def contrastive_loss_2d(objects, gt_mask, num_samples=2000, K=50000, tau=0.6, lambda_neg=0.1, device="cuda"):
    C, H, W = objects.shape
    HW_total = H * W

    feat = objects.view(C, -1).permute(1, 0).to(device) # [C, H, W] -> [HW, C]
    labels = gt_mask.view(-1).to(device) # [HW]

    N = min(num_samples, HW_total)
    indices = torch.randperm(HW_total, device=device)[:N]
    feat_samp = feat[indices] # [N, C]
    labels_samp = labels[indices] # [N]

    # Sample K random pixel pairs
    P1, P2 = torch.randint(0, N, (K,), device=device), torch.randint(0, N, (K,), device=device)
    
    # Remove self-pairs
    valid = (P1 != P2)
    P1, P2 = P1[valid], P2[valid]

    f1, f2 = feat_samp[P1], feat_samp[P2] # [M, C], [M, C]
    L1, L2 = labels_samp[P1], labels_samp[P2] # [M], [M]

    # Cosine similarity between pixel features
    S = cosine_similarity(f1, f2) # [M]

    # Positive loss: maximize similarity for same-object pixels
    same_mask = (L1 == L2)
    if same_mask.any():
        L_pos = - S[same_mask].mean()
    else:
        L_pos = torch.tensor(0.0, device=device)

    # Negative loss: penalize pairs exceeding margin τ
    diff_mask = (L1 != L2)
    if diff_mask.any():
        viol = S[diff_mask] > tau # violations of margin
        if viol.any():
            L_neg = S[diff_mask][viol].mean()
        else:
            L_neg = torch.tensor(0.0, device=device)
    else:
        L_neg = torch.tensor(0.0, device=device)

    loss = L_pos + lambda_neg * L_neg
    return loss


def contrastive_loss_3d(xyz, features, K=5, Ns=2000, max_points=200000, device="cuda"):
    N = xyz.size(0)

    # Optional downsampling for memory stability
    if N > max_points:
        idx = torch.randperm(N, device=device)[:max_points]
        xyz = xyz[idx]
        features = features[idx]
        N = max_points

    # Sample Ns points for computing loss
    Ns = min(Ns, N)
    sample_idx = torch.randperm(N, device=device)[:Ns]

    xyz_sample = xyz[sample_idx]         # [Ns, 3]
    feat_sample = features[sample_idx]   # [Ns, C]

    # Compute distances to all Gaussians
    dists = torch.cdist(xyz_sample, xyz)  # [Ns, N]

    # Get K nearest neighbors (excluding itself)
    knn_dists, knn_idx = torch.topk(dists, K+1, largest=False)
    knn_idx = knn_idx[:, 1:]  # remove self index -> [Ns, K]

    # Gather neighbor features
    neigh_feats = features[knn_idx]  # [Ns, K, C]

    # Compute cosine similarity for each pair
    f_i = feat_sample.unsqueeze(1).expand(-1, K, -1)  # [Ns, K, C]
    cos = cosine_similarity(f_i.reshape(-1, features.shape[1]), neigh_feats.reshape(-1, features.shape[1])).reshape(Ns, K)

    # Final loss (Eq. 12)
    loss = -cos.mean()
    return loss


class SparseDepthLoss:
    def __init__(self, iter_end):
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration <= self.iter_end

    def __call__(self, cam, render_pkg):
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert hasattr(cam, "sparse_pt") and cam.sparse_pt is not None, "No sparse points depth?"
        depth = render_pkg['raw_depth'][0] / (1 - render_pkg['raw_T']).clamp_min_(1e-4)
        sparse_pt = cam.sparse_pt.cuda()
        sparse_uv, sparse_depth = cam.project(sparse_pt, return_depth=True)
        rend_sparse_depth = torch.nn.functional.grid_sample(
            depth[None],
            sparse_uv[None,None],
            mode='bilinear', align_corners=False).squeeze()
        sparse_depth = sparse_depth.squeeze(1)
        return torch.nn.functional.smooth_l1_loss(rend_sparse_depth, sparse_depth)


class DepthAnythingv2Loss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "depthanythingv2"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        invdepth = 1 / render_pkg['raw_depth'].unsqueeze(1).clamp_min(cam.near)
        alpha = (1 - render_pkg['raw_T'][None])
        mono = cam.depthanythingv2.cuda()
        mono = mono[None,None]

        if invdepth.shape[-2:] != mono.shape[-2:]:
            mono = torch.nn.functional.interpolate(
                mono, size=invdepth.shape[-2:], mode='bilinear')

        X, _, Xref = invdepth.split(1)
        X = X * alpha
        Y = mono

        with torch.no_grad():
            Ymed = Y.median()
            Ys = (Y - Ymed).abs().mean()
            Xmed = Xref.median()
            Xs = (Xref - Xmed).abs().mean()
            target = (Y - Ymed) * (Xs/Ys) + Xmed

        mask = (target > 0.01) & (alpha > 0.5)
        X = X * mask
        target = target * mask
        loss = l2_loss(X, target)

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss


class Mast3rMetricDepthLoss:
    def __init__(self, iter_from, iter_end, end_mult):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.end_mult = end_mult

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert hasattr(cam, "mast3r_metric_depth"), "Estimated depth not loaded"
        assert "raw_T" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        alpha = (1 - render_pkg['raw_T'][None])
        depth = render_pkg['raw_depth'][[0]][None]
        ref = cam.mast3r_metric_depth[None,None].cuda()

        if depth.shape[-2:] != ref.shape[-2:]:
            alpha = torch.nn.functional.interpolate(
                alpha, size=ref.shape[-2:], mode='bilinear', antialias=True)
            depth = torch.nn.functional.interpolate(
                depth, size=ref.shape[-2:], mode='bilinear', antialias=True)
            # ref = torch.nn.functional.interpolate(
            #     ref, size=depth.shape[-2:], mode='bilinear')

        # Compute cauchy loss
        active_idx = torch.where(alpha > 0.5)
        depth = depth / alpha
        loss = cauchy_loss(depth[active_idx], ref[active_idx], reduction='sum')
        loss = loss * (1 / depth.numel())

        ratio = (iteration - self.iter_from) / (self.iter_end - self.iter_from)
        mult = self.end_mult ** ratio
        return mult * loss


class NormalDepthConsistencyLoss:
    def __init__(self, iter_from, iter_end, ks, tol_deg):
        self.iter_from = iter_from
        self.iter_end = iter_end
        self.ks = ks
        self.tol_cos = np.cos(np.deg2rad(tol_deg))

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_T" in render_pkg, "Forgot to set `output_T=True` when calling render?"
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # Read rendering results
        render_alpha = 1 - render_pkg['raw_T'].detach().squeeze(0)
        render_depth = render_pkg['raw_depth'][0]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_mean = cam.depth2normal(render_depth, ks=self.ks, tol_cos=self.tol_cos)

        # Blend with alpha and compute target
        target = render_alpha.square()
        N_mean = N_mean * render_alpha

        # Compute loss
        mask = (N_mean != 0).any(0)
        loss_map = (target - (render_normal * N_mean).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss


class NormalMedianConsistencyLoss:
    def __init__(self, iter_from, iter_end):
        self.iter_from = iter_from
        self.iter_end = iter_end

    def is_active(self, iteration):
        return iteration >= self.iter_from and iteration <= self.iter_end

    def __call__(self, cam, render_pkg, iteration):
        assert "raw_depth" in render_pkg, "Forgot to set `output_depth=True` when calling render?"
        assert "raw_normal" in render_pkg, "Forgot to set `output_normal=True` when calling render?"

        if not self.is_active(iteration):
            return 0

        # TODO: median depth is not differentiable
        render_median = render_pkg['raw_depth'][2]
        render_normal = render_pkg['raw_normal']

        # Compute depth to normal
        N_med = cam.depth2normal(render_median, ks=3)

        # Compute loss
        mask = (N_med != 0).any(0)
        loss_map = (1 - (render_normal * N_med).sum(dim=0)) * mask
        loss = loss_map.mean()
        return loss
