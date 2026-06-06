#ifndef RASTERIZER_APPLY_WEIGHTS_H_INCLUDED
#define RASTERIZER_APPLY_WEIGHTS_H_INCLUDED

#include <torch/extension.h>

namespace APPLY_WEIGHTS {

// Interface for python to run apply_weights rasterization.
void rasterize_voxels_apply_weights(
    const int n_samp_per_vox,
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,

    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const torch::Tensor& geos,
    const torch::Tensor& image_weights,  // (C, H, W)
    torch::Tensor& weights,
    torch::Tensor& cnt,

    const torch::Tensor& geomBuffer,

    const bool debug);

}

#endif
