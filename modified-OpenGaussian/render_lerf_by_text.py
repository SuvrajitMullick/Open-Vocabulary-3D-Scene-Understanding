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
import torch.nn as nn
import torch.nn.functional as F
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
import json
from dataclasses import dataclass, field
from typing import Type, Tuple
import open_clip
from utils.opengs_utlis import mask_feature_mean, get_SAM_mask_and_feat, load_code_book

np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)

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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_name):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)

    # load codebook
    root_code_book, root_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "root_code_book"))
    leaf_code_book, leaf_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "leaf_code_book"))
    root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    # counts = torch.bincount(torch.from_numpy(cluster_indices), minlength=64)

    # load the saved codebook(leaf id) and instance-level language feature
    # 'leaf_feat', 'leaf_acore', 'occu_count', 'leaf_ind'
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)
    leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=k1*k2, 512] cluster lang feat
    leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=k1*k2] cluster score
    leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=k1*k2] 
    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] fine id
    leaf_lang_feat[leaf_occu_count < 5] *= 0.0      # Filter out clusters that occur too infrequently.
    leaf_cluster_indices = leaf_ind
    
    # Force cast to integers to prevent range() type errors downstream in the render function
    root_num = int(root_cluster_indices.max().item() + 1)
    leaf_num = int(leaf_lang_feat.shape[0] // root_num)

    # text feature
    with open('assets/text_features.json', 'r') as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]

    scene_texts = {
        "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
                'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
                'pour-over vessel', 'knife', 'yellow desk'],
        "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                'napkin', 'bowl', 'glass of water', 'chopsticks', 'wavy noodles'],
        "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
        "teatime": ['sheep', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
                'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                'bag of cookies']
    }
    # note: query text
    target_text = scene_texts[scene_name]

    query_text_feats = torch.zeros(len(target_text), 512).cuda()
    for i, text in enumerate(target_text):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats[i] = feat
        
    clip_config = OpenCLIPNetworkConfig()
    clip_model = OpenCLIPNetwork(clip_config)

    for t_i, text_feat in enumerate(query_text_feats):
        print(f"rendering the {t_i+1}-th query of {len(target_text)} texts: {target_text[t_i]}")

        positive_prompt = [target_text[t_i]]
        clip_model.set_positives(positive_prompt)
        
        # compute cosine similarity
        text_feat = F.normalize(text_feat.unsqueeze(0), dim=1, p=2)  
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)  
        cosine_similarity = torch.matmul(text_feat, leaf_lang_feat.transpose(0, 1))
        top_values, top_indices = torch.topk(cosine_similarity, 10)
        text_leaf_indices = top_indices[0]

        # render
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            scene_gt_frames = {
                "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
                "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
                "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
                "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
            }
            candidate_frames = scene_gt_frames[scene_name]
            
            if  view.image_name not in candidate_frames:
                continue

            render_pkg = render(view, gaussians, pipeline, background, iteration, rescale=False)
            
            # RGB & Feat Maps
            rendering = render_pkg["render"]
            gt = view.original_image[0:3, :, :]
            rendered_ins_feat = render_pkg["ins_feat"]
            gt_sam_mask = view.original_sam_mask.cuda()    

            # Saves
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendered_ins_feat[:3,:,:], os.path.join(render_ins_feat_path, '{0:05d}'.format(idx) + "_1.png"))
            torchvision.utils.save_image(rendered_ins_feat[3:6,:,:], os.path.join(render_ins_feat_path, '{0:05d}'.format(idx) + "_2.png"))

            # NOTE get SAM id, mask bool, mask_feat, invalid pix
            mask_id, mask_bool, mask_feat, invalid_pix = \
                get_SAM_mask_and_feat(gt_sam_mask, original_mask_feat=view.original_mask_feat)
            
            # sam mask
            mask_color_rand = colors_defined[mask_id.detach().cpu().type(torch.int64)].type(torch.float64)
            mask_color_rand = mask_color_rand.permute(2, 0, 1)
            torchvision.utils.save_image(mask_color_rand/255.0, os.path.join(gt_sam_mask_path, '{0:05d}'.format(idx) + ".png"))
            
            black_background = [0,0,0]
            black_background = torch.tensor(black_background, dtype=torch.float32, device="cuda")
            # render target object
            render_pkg = render(view, gaussians, pipeline, black_background, iteration,
                                rescale=False,                #)  # wherther to re-scale the gaussian scale
                                # cluster_idx=leaf_cluster_indices,     # root id
                                leaf_cluster_idx=leaf_cluster_indices,  # leaf id
                                selected_leaf_id=text_leaf_indices.cuda(),  # text query 所选择的 leaf id
                                render_feat_map=False, 
                                render_cluster=False,
                                seg_rgb=True,
                                root_num=root_num, leaf_num=leaf_num)
            rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
            occured_leaf_id = render_pkg["occured_leaf_id"]
            rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]
            
            cluster_scores_list = []
            for i, img in enumerate(rendered_cluster_imgs):
                leaf_id = int(occured_leaf_id[i])
                img_silhouette = rendered_leaf_cluster_silhouettes[i] > 0.7
                if not img_silhouette.any():
                    continue
                
                y_indices, x_indices = torch.where(img_silhouette.squeeze())
                y_min, y_max = y_indices.min().item(), y_indices.max().item()
                x_min, x_max = x_indices.min().item(), x_indices.max().item()
                if (y_max - y_min) < 5 or (x_max - x_min) < 5:
                    continue
                
                img_crop = img[:3, y_min:y_max, x_min:x_max]
                mask_crop = img_silhouette[y_min:y_max, x_min:x_max].unsqueeze(0)
                masked_crop = img_crop * mask_crop
                
                clip_input = F.interpolate(masked_crop.unsqueeze(0), size=(224, 224), mode='bilinear')
                embed = clip_model.encode_image(clip_input)
                embed /= embed.norm(dim=-1, keepdim=True)

                relevancy_output = clip_model.get_relevancy(embed, 0)
                objectness_score = relevancy_output[0, 0].item()
                cluster_scores_list.append((objectness_score, leaf_id))
            
            cluster_scores_list.sort(key=lambda x: x[0], reverse=True)
            top_clusters = cluster_scores_list[:5]
            top_leaf_indices = []
            for candidate_id in text_leaf_indices:
                candi_feat = leaf_code_book['ins_feat'][candidate_id]
                for _, max_id in top_clusters:
                    max_feat = leaf_code_book['ins_feat'][max_id]
                    distance = torch.norm(max_feat - candi_feat, dim=0)
                    if distance.item() < 0.9:
                        top_leaf_indices.append(candidate_id)
                        break
                
            render_pkg = render(view, gaussians, pipeline, background, iteration,
                                rescale=False,                
                                leaf_cluster_idx=leaf_cluster_indices,
                                selected_leaf_id=text_leaf_indices.cuda(),
                                render_feat_map=False, 
                                render_cluster=False,
                                seg_rgb=True,
                                root_num=root_num, leaf_num=leaf_num)
            rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
            rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]

            render_cluster_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster", view.image_name)
            render_cluster_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cluster_silhouette", view.image_name)
            makedirs(render_cluster_path, exist_ok=True)
            makedirs(render_cluster_silhouette_path, exist_ok=True)

            for i, img in enumerate(rendered_cluster_imgs):
                # save object RGB
                torchvision.utils.save_image(img[:3,:,:], os.path.join(render_cluster_path, \
                    f"{target_text[t_i]}.png"))
                # save object mask
                cluster_silhouette = rendered_leaf_cluster_silhouettes[i] > 0.7
                torchvision.utils.save_image(cluster_silhouette.to(torch.float32), os.path.join(render_cluster_silhouette_path, \
                    f"{target_text[t_i]}.png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                scene_name: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "text2obj_train_new", scene.loaded_iter, scene.getTrainCameras(), 
                        gaussians, pipeline, background, scene_name)
        if not skip_test:
             render_set(dataset.model_path, "text2obj_test_new", scene.loaded_iter, scene.getTestCameras(), 
                        gaussians, pipeline, background, scene_name)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scene_name", type=str, choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
                        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not args.scene_name:
        parser.error("The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scene_name)