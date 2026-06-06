conda env create -f environment.yml
conda activate svraster

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

pip install --no-build-isolation ./cuda/
pip install --no-build-isolation ./fused-ssim/
pip install --no-build-isolation ./GroundingDINO/
pip install --no-build-isolation ./sam-hq/



bash scripts/train_render_eval.sh ramen output_sam object_mask_sam normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam crop_object_mask_sam normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified object_mask_sam_unified normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_unified crop_object_mask_sam_unified normal 1.5
bash scripts/train_render_eval.sh ramen output_sam_hq object_mask_sam_hq normal 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq crop_object_mask_sam_hq normal 1.5


bash scripts/train_render_eval.sh ramen output_sam_unified_cos object_mask_sam_unified cos 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified_hyp object_mask_sam_unified hyp 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified_vst object_mask_sam_unified vst 1.5
bash scripts/train_render_eval.sh ramen output_sam_unified_all object_mask_sam_unified all 1.5
 

bash scripts/train_render_eval.sh ramen output_crop_sam_hq_cos crop_object_mask_sam_hq cos 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_vst crop_object_mask_sam_hq vst 1.5
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_all crop_object_mask_sam_hq all 1.5




bash scripts/train_render_eval.sh figurines output_sam object_mask_sam normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam crop_object_mask_sam normal 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified object_mask_sam_unified normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_unified crop_object_mask_sam_unified normal 0.025
bash scripts/train_render_eval.sh figurines output_sam_hq object_mask_sam_hq normal 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq crop_object_mask_sam_hq normal 0.025


bash scripts/train_render_eval.sh figurines output_sam_unified_cos object_mask_sam_unified cos 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified_hyp object_mask_sam_unified hyp 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified_vst object_mask_sam_unified vst 0.025
bash scripts/train_render_eval.sh figurines output_sam_unified_all object_mask_sam_unified all 0.025
 

bash scripts/train_render_eval.sh figurines output_crop_sam_hq_cos crop_object_mask_sam_hq cos 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_vst crop_object_mask_sam_hq vst 0.025
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_all crop_object_mask_sam_hq all 0.025




bash scripts/train_render_eval.sh teatime output_sam object_mask_sam normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam crop_object_mask_sam normal 1
bash scripts/train_render_eval.sh teatime output_sam_unified object_mask_sam_unified normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam_unified crop_object_mask_sam_unified normal 1
bash scripts/train_render_eval.sh teatime output_sam_hq object_mask_sam_hq normal 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq crop_object_mask_sam_hq normal 1


bash scripts/train_render_eval.sh teatime output_sam_unified_cos object_mask_sam_unified cos 1
bash scripts/train_render_eval.sh teatime output_sam_unified_hyp object_mask_sam_unified hyp 1
bash scripts/train_render_eval.sh teatime output_sam_unified_vst object_mask_sam_unified vst 1
bash scripts/train_render_eval.sh teatime output_sam_unified_all object_mask_sam_unified all 1


bash scripts/train_render_eval.sh teatime output_crop_sam_hq_cos crop_object_mask_sam_hq cos 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_hyp crop_object_mask_sam_hq hyp 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_vst crop_object_mask_sam_hq vst 1
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_all crop_object_mask_sam_hq all 1



