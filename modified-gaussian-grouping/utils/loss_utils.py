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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def loss_cls_3d(features, predictions, k=5, max_points=200000, sample_size=2000):

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


def multiclass_iou_loss(probs, target, eps=1e-6):
    # One-hot target
    target_one_hot = F.one_hot(target, num_classes=probs.shape[0]) # (H, W, C)
    target_one_hot = target_one_hot.permute(2,0,1).float() # (H, W, C) → (C, H, W)

    # Intersection & union per class
    intersection = (probs * target_one_hot).sum(dim=(1,2)) # (C,)
    union = (probs + target_one_hot - probs * target_one_hot).sum(dim=(1,2)) + eps # (C,)

    # IoU and loss
    iou = intersection / union
    return 1 - iou.mean()

def multiclass_iou_loss(prob_obj_2d, label, eps=1e-9):
    unique_classes = label.unique().tolist() # Only compute IoU for present classes
    intersection_list = []
    union_list = []

    for cls in unique_classes:
        gt_mask = (label == cls) # (H, W) boolean
        pred_mask = prob_obj_2d[cls] # (H, W)

        intersection = pred_mask[gt_mask].sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection + eps

        intersection_list.append(intersection)
        union_list.append(union)

    intersection_all = torch.stack(intersection_list)
    union_all = torch.stack(union_list)

    iou_per_class = intersection_all / union_all
    mean_iou = iou_per_class.mean()
    return mean_iou


'''
            for idx in view_indices:
                cam = colmap_cameras[idx]
                label = cam.objects.cuda().long() # label: [H,W] grayscale label map
                
                image_weights = torch.nn.functional.one_hot(label, num_classes=num_classes) # one-hot: [H, W, C]
                image_weights = image_weights.permute(2, 0, 1).float().contiguous() # (H, W, C) → (C, H, W)

                gaussians.apply_weights(cam, weights, weights_cnt, image_weights)         
'''
        
'''

    criterion_p = PerceptualLoss().eval().to("cuda")
        
        # Perceptual Image Loss
        percept_image_loss = criterion_p(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
        loss += opt.perceptual_lambda_image * percept_image_loss
        # print(percept_image_loss)
'''

'''
        print("image shape:", image.shape)
        print("gt_image shape:", gt_image.shape)
        print("objects shape:", objects.shape)
        print("gt_obj shape:", gt_obj.shape)
        
        print("image:", image)
        print("gt_image:", gt_image)
        print("objects:", objects)
        print("gt_obj:", gt_obj)
'''

'''
        # IoU Object Losses
        iou_loss = None
        # regularize at certain intervals
        if iteration % opt.iou_interval == 0:
            colmap_cameras = scene.getTrainCameras().copy()
            iou_loss = 0

            total_view_num = len(colmap_cameras)
            view_indices = random.sample(range(total_view_num), min(total_view_num, opt.num_of_views))
            
            for idx in view_indices:
                cam = colmap_cameras[idx]
                label = cam.objects.cuda().long() # label: [H,W] grayscale label map

                render_pkg = render(cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                objects = render_pkg["render_object"]
                
                logits_2d = classifier(objects)
                prob_obj_2d = torch.softmax(logits_2d, dim=0) # (C, H, W)
                iou = multiclass_iou_loss(prob_obj_2d, label)
                iou_loss += (1 - iou)
                
            # average IoU over all views
            mean_iou_loss = iou_loss / opt.num_of_views
            loss += opt.lambda_iou * mean_iou_loss
            # print(mean_iou_loss)
'''

'''
            print("Ll1:", Ll1, "lambda multiplied value:", (1.0 - opt.lambda_dssim) * Ll1)
            print("1.0 - ssim_value:", 1.0 - ssim_value, "lambda multiplied value:", opt.lambda_dssim * (1.0 - ssim_value))
            print("loss_3d_hypersphere:", loss_3d_hypersphere, "lambda multiplied value:", opt.hypersphere_lambda_3d * loss_3d_hypersphere)
            print("loss_2d_hypersphere:", loss_2d_hypersphere, "lambda multiplied value:", opt.hypersphere_lambda_2d * loss_2d_hypersphere)
            print("loss_obj_2d:", loss_obj_2d, "lambda multiplied value:", opt.reg2d_lambda_val * loss_obj_2d)
            print("loss_obj_3d:", loss_obj_3d, "lambda multiplied value:", opt.reg3d_lambda_val * loss_obj_3d)
            print("contrast_loss_2d:", contrast_loss_2d, "lambda multiplied value:", opt.contrastive_lambda_2d * contrast_loss_2d)
            print("contrast_loss_3d:", contrast_loss_3d, "lambda multiplied value:", opt.contrastive_lambda_3d * contrast_loss_3d)
            print("loss_obj_wt_3d:", loss_obj_wt_3d, "lambda multiplied value:", opt.reg3d_wt_lambda_val * loss_obj_wt_3d)
'''