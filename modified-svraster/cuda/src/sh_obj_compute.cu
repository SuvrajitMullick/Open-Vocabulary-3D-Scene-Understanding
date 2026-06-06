/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "sh_obj_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

struct float16
{
    float d[16];
};

typedef struct float16 float16;

namespace SH_OBJ_COMPUTE {

__device__ const float SH_C0 = 0.28209479177387814f;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple OBJ objects.
__device__ float16 computeObjectFromSHOBJ(
    int idx,
    const float16* sh_objs)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)

    // float16 result = SH_C0 * sh_objs[idx];
    // result = result + 0.5f;
    
    float16 result;
    #pragma unroll
    for (int k = 0; k < 16; ++k)
        result.d[k] = SH_C0 * sh_objs[idx].d[k];

    // OBJ objects are clamped to non-negative values.
    // #pragma unroll
    // for (int k = 0; k < 16; ++k)
    //     result.d[k] *= (result.d[k] > 0.0f);
    return result;
}


// Backward pass for spherical harmonics to OBJ.
__device__ void computeObjectFromSHOBJ_bw(
    int idx,
    const float16 obj, float16 dL_dobj, float16* dL_dsh_objs)
{
    // Check if the object was clampped in the forward pass.
    // #pragma unroll
    // for (int k = 0; k < 16; ++k)
    //     dL_dobj.d[k] *= (float)(obj.d[k] > 0);

    // Adapt the implementation from 3DGS.
    float dOBJdsh_objs = SH_C0;
    // dL_dsh_objs[idx] = dOBJdsh_objs * dL_dobj;
    #pragma unroll
    for (int k = 0; k < 16; ++k)
        dL_dsh_objs[idx].d[k] = dOBJdsh_objs * dL_dobj.d[k];

}


// Compute obj from spherical harmonic.
__global__ void sh_obj_compute_cuda(
    const int N, const int n_vox,
    const int64_t* __restrict__ indices,
    const float16* __restrict__ sh_objs,
    float16* __restrict__ objs)
{
    auto tid = cg::this_grid().thread_rank();
    if ((N == 0 && tid >= n_vox) || (N != 0 && tid >= N))
        return;

    // Load from global memory.
    const int idx = (N != 0) ? indices[tid] : tid;

    // Convert spherical harmonics coefficients to OBJ object.
    auto sh_obj_eval = computeObjectFromSHOBJ;
    float16 sh_obj_result = sh_obj_eval(idx, sh_objs);

    // Write back the results.
    objs[idx] = sh_obj_result;
}


// Backward pass of the preprocessing steps.
__global__ void sh_obj_compute_bw_cuda(
    const int N, const int n_vox,
    const int64_t* __restrict__ indices,
    const float16* __restrict__ objs,
    const float16* __restrict__ dL_dobjs,
    float16* __restrict__ dL_dsh_objs)
{
    auto tid = cg::this_grid().thread_rank();
    if ((N == 0 && tid >= n_vox) || (N != 0 && tid >= N))
        return;

    // Load from global memory.
    const int idx = (N != 0) ? indices[tid] : tid;
    const float16 obj = objs[idx];
    const float16 dL_dobj = dL_dobjs[idx];

    // Compute gradient updates due to computing objects from SHOBJs
    auto sh_eval = computeObjectFromSHOBJ_bw;
    sh_eval(idx, obj, dL_dobj, dL_dsh_objs);
}


// Python interface for spherical harmonic computation.
torch::Tensor sh_obj_compute(
    const torch::Tensor& indices,
    const torch::Tensor& vox_centers,
    const torch::Tensor& sh_objs)
{
    const int P = vox_centers.size(0);
    const int N = indices.size(0);
    torch::Tensor objs = torch::zeros({P, 16}, vox_centers.options());

    const int total_threads = N != 0 ? N : P;

    if (P > 0)
        sh_obj_compute_cuda <<<(total_threads + 255) / 256, 256>>> (
            N, P,
            indices.contiguous().data_ptr<int64_t>(),
            (float16*)sh_objs.contiguous().data_ptr<float>(),
            (float16*)objs.contiguous().data_ptr<float>());
    
    return objs;
}

torch::Tensor sh_obj_compute_bw(
    const torch::Tensor& indices,
    const torch::Tensor& vox_centers,
    const torch::Tensor& objs,
    const torch::Tensor& dL_dobjs)
{
    const int P = vox_centers.size(0);
    const int N = indices.size(0);
    torch::Tensor dL_dsh_objs = torch::zeros({P, 16}, vox_centers.options());
    // torch::Tensor dL_dshs = torch::zeros({P, M-1, 3}, vox_centers.options());

    const int total_threads = N != 0 ? N : P;

    if (P > 0)
        sh_obj_compute_bw_cuda <<<(total_threads + 255) / 256, 256>>> (
            N, P,
            indices.contiguous().data_ptr<int64_t>(),
            (float16*)objs.contiguous().data_ptr<float>(),
            (float16*)dL_dobjs.contiguous().data_ptr<float>(),
            (float16*)dL_dsh_objs.contiguous().data_ptr<float>());

    return dL_dsh_objs;
}

}
