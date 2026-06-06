cd OpenGaussian
conda env create -f environment.yml
conda activate open_gaussian

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install ninja
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

pip install --no-build-isolation submodules/ashawkey-diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation submodules/sam-langsplat
pip install --no-build-isolation submodules/sam-hq

cd assets
unzip text_features.zip
cd ..



python preprocess_sam_l.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_l.py --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/ramen --crop
python preprocess_sam_u.py --dataset_path data/lerf_ovs/ramen
python preprocess_sam_u.py --dataset_path data/lerf_ovs/ramen --crop
# python preprocess_sam_u_vis.py --dataset_path data/lerf_ovs/ramen --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/ramen --crop --variant sam_u


python preprocess_sam_l.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_l.py --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/figurines --crop
python preprocess_sam_u.py --dataset_path data/lerf_ovs/figurines
python preprocess_sam_u.py --dataset_path data/lerf_ovs/figurines --crop
# python preprocess_sam_u_vis.py --dataset_path data/lerf_ovs/figurines --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/figurines --crop --variant sam_u


python preprocess_sam_l.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_l.py --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_hq.py --dataset_path data/lerf_ovs/teatime --crop
python preprocess_sam_u.py --dataset_path data/lerf_ovs/teatime
python preprocess_sam_u.py --dataset_path data/lerf_ovs/teatime --crop
# python preprocess_sam_u_vis.py --dataset_path data/lerf_ovs/teatime --crop

python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_l
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_hq
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --variant sam_u
python masks_visualizer.py --dataset_path data/lerf_ovs/teatime --crop --variant sam_u




python train_normal.py -s data/lerf_ovs/ramen -m output_full_scene/ramen --iterations 30_000
cp -r output_full_scene/ramen output/ramen
python crop_scene.py -m output/ramen --iteration 30000 --padding 1.5
python crop_images.py -m output/ramen --iteration 30000

cp -r output_full_scene/ramen output_sam_l_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_l_full_scene language_features_sam_l normal chkpnt30000

cp -r output_full_scene/ramen output_crop_sam_l_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

cp -r output/ramen output_crop_sam_l/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

cp -r output_full_scene/ramen output_sam_hq_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/ramen output_crop_sam_hq_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

cp -r output/ramen output_crop_sam_hq/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/ramen output_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_sam_u_full_scene language_features_sam_u normal chkpnt30000

cp -r output_full_scene/ramen output_crop_sam_u_full_scene/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

cp -r output/ramen output_crop_sam_u/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u crop_language_features_sam_u normal chkpnt30000


python render_lerf_by_text.py -m output_crop_sam_u/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/ramen --split train

cp -r output_crop_sam_u/ramen output_crop_sam_u_prune/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_prune/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/ramen --split train

cp -r output_crop_sam_u/ramen output_crop_sam_u_no_filter/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_no_filter/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/ramen --split train

cp -r output_crop_sam_u/ramen output_crop_sam_u_all/ramen
bash scripts/train_render_eval.sh ramen output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
python render_lerf_by_text.py -m output_crop_sam_u_all/ramen --scene_name ramen --skip_test
python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/ramen --split train




python train_normal.py -s data/lerf_ovs/figurines -m output_full_scene/figurines --iterations 30_000
cp -r output_full_scene/figurines output/figurines
python crop_scene.py -m output/figurines --iteration 30000 --padding 0.025
python crop_images.py -m output/figurines --iteration 30000

cp -r output_full_scene/figurines output_sam_l_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_l_full_scene language_features_sam_l normal chkpnt30000

cp -r output_full_scene/figurines output_crop_sam_l_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

cp -r output/figurines output_crop_sam_l/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

cp -r output_full_scene/figurines output_sam_hq_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/figurines output_crop_sam_hq_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

cp -r output/figurines output_crop_sam_hq/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/figurines output_sam_u_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_sam_u_full_scene language_features_sam_u normal chkpnt30000

cp -r output_full_scene/figurines output_crop_sam_u_full_scene/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

cp -r output/figurines output_crop_sam_u/figurines
bash scripts/train_render_eval.sh figurines output_crop_sam_u crop_language_features_sam_u normal chkpnt30000


# python render_lerf_by_text.py -m output_crop_sam_u/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_prune/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_prune/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_no_filter/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_no_filter/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/figurines --split train

# cp -r output_crop_sam_u/figurines output_crop_sam_u_all/figurines
# bash scripts/train_render_eval.sh figurines output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_all/figurines --scene_name figurines --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/figurines --split train




python train_normal.py -s data/lerf_ovs/teatime -m output_full_scene/teatime --iterations 30_000
cp -r output_full_scene/teatime output/teatime
python crop_scene.py -m output/teatime --iteration 30000 --padding 1
python crop_images.py -m output/teatime --iteration 30000

cp -r output_full_scene/teatime output_sam_l_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_l_full_scene language_features_sam_l normal chkpnt30000

cp -r output_full_scene/teatime output_crop_sam_l_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_l_full_scene crop_language_features_sam_l normal chkpnt30000

cp -r output/teatime output_crop_sam_l/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_l crop_language_features_sam_l normal chkpnt30000

cp -r output_full_scene/teatime output_sam_hq_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_hq_full_scene language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/teatime output_crop_sam_hq_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_hq_full_scene crop_language_features_sam_hq normal chkpnt30000

cp -r output/teatime output_crop_sam_hq/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_hq crop_language_features_sam_hq normal chkpnt30000

cp -r output_full_scene/teatime output_sam_u_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_sam_u_full_scene language_features_sam_u normal chkpnt30000

cp -r output_full_scene/teatime output_crop_sam_u_full_scene/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_u_full_scene crop_language_features_sam_u normal chkpnt30000

cp -r output/teatime output_crop_sam_u/teatime
bash scripts/train_render_eval.sh teatime output_crop_sam_u crop_language_features_sam_u normal chkpnt30000


# python render_lerf_by_text.py -m output_crop_sam_u/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_prune/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_prune crop_language_features_sam_u prune chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_prune/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_prune/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_no_filter/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_no_filter crop_language_features_sam_u no_filter chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_no_filter/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_no_filter/teatime --split train

# cp -r output_crop_sam_u/teatime output_crop_sam_u_all/teatime
# bash scripts/train_render_eval.sh teatime output_crop_sam_u_all crop_language_features_sam_u all chkpnt40000
# python render_lerf_by_text.py -m output_crop_sam_u_all/teatime --scene_name teatime --skip_test
# python scripts/eval_lerf_mask_new.py --out_dir output_crop_sam_u_all/teatime --split train



