mkdir -P checkpoints
wget -P checkpoints/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P checkpoints/ https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
wget -P checkpoints/ https://huggingface.co/likeabruh/depth_anything_v2_vitg/resolve/main/depth_anything_v2_vitg.pth
wget -P checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
