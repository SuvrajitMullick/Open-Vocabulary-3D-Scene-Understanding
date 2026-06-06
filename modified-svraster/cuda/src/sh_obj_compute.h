/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef SH_OBJ_COMPUTE_H_INCLUDED
#define SH_OBJ_COMPUTE_H_INCLUDED

#include <torch/extension.h>

namespace SH_OBJ_COMPUTE {

// Python interface for spherical harmonic computation.
torch::Tensor sh_obj_compute(
    const torch::Tensor& idx,
    const torch::Tensor& vox_centers,
    const torch::Tensor& sh_objs);

torch::Tensor sh_obj_compute_bw(
    const torch::Tensor& idx,
    const torch::Tensor& vox_centers,
    const torch::Tensor& objs,
    const torch::Tensor& dL_dobjs);

}

#endif
