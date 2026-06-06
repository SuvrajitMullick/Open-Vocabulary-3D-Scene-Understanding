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

#include "apply_weights.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace APPLY_WEIGHTS {

// CUDA sparse voxel rendering.
template <int CHANNELS,
			int n_samp>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_apply_weights(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ vox_list,
    int W, int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* __restrict__ c2w_matrix,

    const uint2* __restrict__ bboxes,
    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,
    const float* __restrict__ geos,

    float* __restrict__ weights,           // (P * CHANNELS)
    float* __restrict__ cnt,               // (P * CHANNELS)
    const float* __restrict__ image_weights, // (CHANNELS * H * W)

    uint32_t* __restrict__ tile_last,
    uint32_t* __restrict__ n_contrib)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    int thread_id = block.thread_rank();
    int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

    uint2 pix;
    uint32_t pix_id;
    float2 pixf;
    if (BLOCK_X % 8 == 0 && BLOCK_Y % 4 == 0)
    {
        // Pack the warp threads into a 4x8 macro blocks.
        // It could reduce idle warp threads as the voxels to render
        // are more coherent in 4x8 than 2x16 rectangle.
        int macro_x_num = BLOCK_X / 8;
        int macro_id = thread_id / 32;
        int macro_xid = macro_id % macro_x_num;
        int macro_yid = macro_id / macro_x_num;
        int micro_id = thread_id % 32;
        int micro_xid = micro_id % 8;
        int micro_yid = micro_id / 8;
        pix = { pix_min.x + macro_xid * 8 + micro_xid, pix_min.y + macro_yid * 4 + micro_yid};
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }
    else
    {
        pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }

    // Compute camera info.
    float3 ro, rd, rd_inv;
    float rd_norm_inv;
    const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
    const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
    const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
    rd_norm_inv = 1.f / rd_norm;
    ro = last_col_3x4(c2w_matrix);
    rd = rd_raw * rd_norm_inv;
    rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};

    const uint32_t pix_quad_id = compute_ray_quadrant_id(rd);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = (pix.x < W) && (pix.y < H);
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in BinningState.
    uint2 range = ranges[tile_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;
	
    // Init the last non-occluded range index of the tile.
    if (thread_id == 0)
        tile_last[tile_id] = range.x;

    // Preload per-pixel channel weights.
    float pix_w[CHANNELS] = {0};
    if (inside)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
            pix_w[ch] = image_weights[ch * H * W + pix_id];
    }

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_vox_id[BLOCK_SIZE];
    __shared__ int collected_quad_id[BLOCK_SIZE];
    __shared__ uint2 collected_bbox[BLOCK_SIZE];
    __shared__ float3 collected_vox_c[BLOCK_SIZE];
    __shared__ float collected_vox_l[BLOCK_SIZE];
    __shared__ float collected_geo_params[BLOCK_SIZE * 8];

    // Initialize helper variables.
    float T = 1.f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    int j_lst[BLOCK_SIZE];
    
    // Iterate over batches until all done or range is complete.
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing.
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch batch of voxel data from global to shared.
        int progress = i * BLOCK_SIZE + thread_id;
        if (range.x + progress < range.y)
        {
            uint32_t order_val = vox_list[range.x + progress];
            uint32_t vox_id = decode_order_val_4_vox_id(order_val);
            uint32_t quad_id = decode_order_val_4_quadrant_id(order_val);
            collected_vox_id[thread_id] = vox_id;
            collected_quad_id[thread_id] = quad_id;
            collected_bbox[thread_id] = bboxes[vox_id];
            collected_vox_c[thread_id] = vox_centers[vox_id];
            collected_vox_l[thread_id] = vox_lengths[vox_id];
            for (int k=0; k<8; ++k)
                collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
        }
        block.sync();
        
        // Iterate over current batch.
        const int end_j = min(BLOCK_SIZE, toDo);
        int j_lst_top = -1;
        for (int j = 0; !done && j < end_j; j++)
        {
            // Check if the pixel in the projected bbox region.
            // Check if the quadrant id match the pixel.
            if (!pix_in_bbox(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
                continue;
            
            // Compute ray aabb intersection
            const float3 vox_c = collected_vox_c[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;
            if (a > b)
                continue;  // Skip if no intersection.
            
            j_lst_top += 1;
            j_lst[j_lst_top] = j;
        }
            
        int contributor_inc = 0;
        for (int jj = 0; !done && jj <= j_lst_top; jj++)
        {
            int j = j_lst[jj];
            const int vox_id = collected_vox_id[j];
            
            // Keep track of current position in range.
            contributor_inc = j + 1;
            
            // Compute ray aabb intersection
            const float3 vox_c = collected_vox_c[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;

            float geo_params[8];
            for (int k=0; k<8; ++k)
                geo_params[k] = collected_geo_params[j*8 + k];

            // Compute volume density
            float vol_int = 0.f;
            float interp_w[8];

            // Quadrature integral from trilinear sampling.
            float vox_l_inv = 1.f / vox_l;
            const float step_sz = (b - a) * (1.f / n_samp);
            const float3 step = step_sz * rd;
            float3 pt = ro + (a + 0.5f * step_sz) * rd;
            float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
            const float3 qt_step = step * vox_l_inv;

            #pragma unroll
            for (int k=0; k<n_samp; k++, qt=qt+qt_step)
            {
                tri_interp_weight(qt, interp_w);
                float d = 0.f;
                for (int iii=0; iii<8; ++iii)
                    d += geo_params[iii] * interp_w[iii];

                const float local_vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
                vol_int += local_vol_int;
            }

            // Compute alpha from volume integral.
            float alpha = min(MAX_ALPHA, 1.f - expf(-vol_int));
            if (alpha < MIN_ALPHA)
                continue;

            // Contribution of weights
            // Accumulate per-pixel weights back to per-voxel weights & counts.
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                atomicAdd(&(weights[vox_id * CHANNELS + ch]), pix_w[ch] * T);
                atomicAdd(&(cnt[vox_id * CHANNELS + ch]), pix_w[ch]);
            }

            T *= (1.f - alpha);
            done |= (T < EARLY_STOP_T);

            // Keep track of last range entry to update this pixel.
            last_contributor = contributor + contributor_inc;
        }
        contributor += done ? contributor_inc : end_j;
    }

    if (inside)
	{
        n_contrib[pix_id] = last_contributor;
        atomicMax(tile_last + tile_id, range.x + last_contributor);
	}
}

// Lowest-level C interface for launching the CUDA.
void render(
    const dim3 tile_grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* vox_list,
    const int n_samp_per_vox,
    int W, int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* c2w_matrix,

    const uint2* bboxes,
    const float3* vox_centers,
    const float* vox_lengths,
    const float* geos,

    float* weights,
    float* cnt,
    const float* image_weights,

    uint32_t* tile_last,
    uint32_t* n_contrib,
    int num_channels)
{
    if (num_channels == 1)
    {
        if (n_samp_per_vox == 3)
            renderCUDA_apply_weights<1,3><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else if (n_samp_per_vox == 2)
            renderCUDA_apply_weights<1,2><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else
            renderCUDA_apply_weights<1,1><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
    }
    else if (num_channels == 2)
    {
        if (n_samp_per_vox == 3)
            renderCUDA_apply_weights<2,3><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else if (n_samp_per_vox == 2)
            renderCUDA_apply_weights<2,2><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else
            renderCUDA_apply_weights<2,1><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
    }
    else if (num_channels == 3)
    {
        if (n_samp_per_vox == 3)
            renderCUDA_apply_weights<3,3><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else if (n_samp_per_vox == 2)
            renderCUDA_apply_weights<3,2><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else
            renderCUDA_apply_weights<3,1><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
    }
	else if (num_channels == 256)
    {
        if (n_samp_per_vox == 3)
            renderCUDA_apply_weights<256,3><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else if (n_samp_per_vox == 2)
            renderCUDA_apply_weights<256,2><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,

                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
        else
            renderCUDA_apply_weights<256,1><<<tile_grid, block>>>(
                ranges,
				vox_list,
				W, H,
                tan_fovx, tan_fovy,
				cx, cy,
				c2w_matrix,
				
                bboxes,
				vox_centers,
				vox_lengths,
				geos,
                weights,
				cnt,
				image_weights,
				
        		tile_last,
				n_contrib);
    }
    else
    {
        printf("Unsupported number of channels: %d\n", num_channels);
        exit(-1);
    }
}


// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Duplicate each voxel by #tiles x #cam_quadrant it touches.
__global__ void duplicateWithKeys(
    int P,
    const int64_t* octree_paths,
    const uint2* bboxes,
    const uint32_t* cam_quadrant_bitsets,
    const uint32_t* n_duplicates,
    const uint32_t* n_duplicates_scan,
    uint64_t* vox_list_keys_unsorted,
    uint32_t* vox_list_unsorted,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || n_duplicates[idx] == 0)
        return;

    // Find this voxel's array offset in buffer for writing the key/value.
    uint32_t off = (idx == 0) ? 0 : n_duplicates_scan[idx - 1];
    uint2 tile_min, tile_max;
    getBboxTileRect(bboxes[idx], tile_min, tile_max, grid);

    // For each tile that the bounding rect overlaps, emit a key/value pair.
    // The key bit structure is [  tile ID  |  order_rank  ],
    // so the voxels are first sorted by tile and then by order_ranks.
    // The value bit structure is [  quadrant ID  |  voxel ID  ].
    const uint64_t octree_path = octree_paths[idx];
    uint32_t quadrant_bitsets = cam_quadrant_bitsets[idx];
    for (int quadrant_id = 0; quadrant_id < 8; quadrant_id++)
    {
        if ((quadrant_bitsets & (1 << quadrant_id)) == 0)
            continue;

        // Compute order_rank for the voxel in this quadrant.
        uint64_t order_rank = compute_order_rank(octree_path, quadrant_id);

        // Duplicate result to touched tiles.
        for (int y = tile_min.y; y <= tile_max.y; y++)
        {
            for (int x = tile_min.x; x <= tile_max.x; x++)
            {
                uint64_t tile_id = y * grid.x + x;
                vox_list_keys_unsorted[off] = encode_order_key(tile_id, order_rank);
                vox_list_unsorted[off] = encode_order_val(idx, quadrant_id);
                off++;
            }
        }
    }

    if (off != n_duplicates_scan[idx])
    {
        // TODO: remove sanity check.
        printf("Number of duplication mismatch !???");
        __trap();
    }
}

// The sorted vox_list_keys is now as:
//   [--sorted voxels for tile 1--  --sorted voxels for tile 2--  ...]
// We want to identify the start/end index of each tile from this list.
__global__ void identifyTileRanges(int L, uint64_t* vox_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = vox_list_keys[idx];
    uint32_t currtile = key >> NUM_BIT_ORDER_RANK;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = vox_list_keys[idx - 1] >> NUM_BIT_ORDER_RANK;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

// Mid-level C interface for the apply-weights rasterization procedure.
void rasterize_voxels_apply_weights_procedure(
    char* geom_buffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P,
    const int n_samp_per_vox,
    const int width, const int height,
    const float tan_fovx, const float tan_fovy,
    const float cx, float cy,
    const float* w2c_matrix,
    const float* c2w_matrix,

    const int64_t* octree_paths,
    const float* vox_centers,
    const float* vox_lengths,
    const float* geos,
	
    const float* image_weights,
    float* weights,
    float* cnt,
    const int num_channels,

    bool debug)
{
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Recover the preprocessing results.
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(geom_buffer, P);

    // Dynamically resize image-based auxiliary buffers during training.
    size_t img_chunk_size = RASTER_STATE::required<RASTER_STATE::ImageState>(width * height, tile_grid.x * tile_grid.y);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    RASTER_STATE::ImageState imgState = RASTER_STATE::ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);

    // Compute prefix sum over full list of the number of voxel duplications.
    cub::DeviceScan::InclusiveSum(
        geomState.scanning_temp_space,
        geomState.scan_size,
        geomState.n_duplicates,
        geomState.n_duplicates_scan,
        P);
    CHECK_CUDA(debug);

    // Retrieve total number of voxels after duplication.
    int num_rendered;
    cudaMemcpy(
        &num_rendered,
        geomState.n_duplicates_scan + P - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug);

    size_t binning_chunk_size = RASTER_STATE::required<RASTER_STATE::BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    RASTER_STATE::BinningState binningState = RASTER_STATE::BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each voxel to be rendered, produce adequate [ tile ID | rank ] key
    // and the corresponding dublicated voxel [ quadrant ID | voxel ID ] to be sorted.
    duplicateWithKeys <<<(P + 255) / 256, 256>>> (
        P,
        octree_paths,
        geomState.bboxes,
        geomState.cam_quadrant_bitsets,
        geomState.n_duplicates,
        geomState.n_duplicates_scan,
        binningState.vox_list_keys_unsorted,
        binningState.vox_list_unsorted,
        tile_grid);
    CHECK_CUDA(debug);

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) ID by keys.
    cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.vox_list_keys_unsorted, binningState.vox_list_keys,
        binningState.vox_list_unsorted, binningState.vox_list,
        num_rendered, 0, NUM_BIT_ORDER_RANK + bit);
    CHECK_CUDA(debug);

    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
    CHECK_CUDA(debug);

    // Identify start and end of per-tile workloads in sorted list.
    if (num_rendered > 0)
    {
        identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
            num_rendered,
            binningState.vox_list_keys,
            imgState.ranges);
        CHECK_CUDA(debug);
    }

    // Let each tile accumulate weights for its range of voxels.
    render(
        tile_grid, block,
        imgState.ranges,
        binningState.vox_list,
        n_samp_per_vox,
        width, height,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix,

        geomState.bboxes,
        (float3*)vox_centers,
        vox_lengths,
        geos,

        weights,
        cnt,
        image_weights,

        imgState.tile_last,
        imgState.n_contrib,
        num_channels);
    CHECK_CUDA(debug);
}


// Interface for python to run apply-weights rasterization.
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

    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");
    if (image_weights.ndimension() != 3)
        AT_ERROR("image_weights must have dimensions (C, H, W)");

    const int P = vox_centers.size(0);
    const int H = image_height;
    const int W = image_width;

  	const int num_channels = image_weights.size(0);

    if (image_weights.size(1) != H || image_weights.size(2) != W)
        AT_ERROR("image_weights height/width must match image_height/image_width");

    auto float_opts = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    auto byte_opts = torch::TensorOptions(torch::kByte).device(torch::kCUDA);

    torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
    torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
    std::function<char*(size_t)> binningFunc = RASTER_STATE::resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = RASTER_STATE::resizeFunctional(imgBuffer);

    int rendered = 0;
    if(P != 0)
        rasterize_voxels_apply_weights_procedure(
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            binningFunc,
            imgFunc,
            P,
            n_samp_per_vox,

            W, H,
            tan_fovx, tan_fovy,
            cx, cy,
            w2c_matrix.contiguous().data_ptr<float>(),
            c2w_matrix.contiguous().data_ptr<float>(),

            octree_paths.contiguous().data_ptr<int64_t>(),
            vox_centers.contiguous().data_ptr<float>(),
            vox_lengths.contiguous().data_ptr<float>(),
            geos.contiguous().data_ptr<float>(),
            image_weights.contiguous().data_ptr<float>(),
			
            weights.contiguous().data_ptr<float>(),
            cnt.contiguous().data_ptr<float>(),
            num_channels,

            debug);

}

}
