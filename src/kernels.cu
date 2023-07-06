/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transfusion/kernels.hpp"

namespace transfusion {
__global__ void generateVoxels_random_kernel(
    float *points, unsigned int points_size, float min_x_range, float max_x_range,
    float min_y_range, float max_y_range, float min_z_range, float max_z_range,
    float pillar_x_size, float pillar_y_size, float pillar_z_size,
    int grid_y_size, int grid_x_size, unsigned int *mask, float *voxels) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size)
    return;

  // float4 point = ((float4 *)points)[point_idx];
  float x = points[point_idx * 5];
  float y = points[point_idx * 5 + 1];
  float z = points[point_idx * 5 + 2];
  float w = points[point_idx * 5 + 3];
  float t = points[point_idx * 5 + 4];

  if (x < min_x_range || x >= max_x_range ||
      y < min_y_range || y >= max_y_range ||
      z < min_z_range || z >= max_z_range)
    return;

  int voxel_idx = floorf((x - min_x_range) / pillar_x_size);
  int voxel_idy = floorf((y - min_y_range) / pillar_y_size);
  unsigned int voxel_index = voxel_idy * grid_x_size + voxel_idx;

  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);

  if (point_id >= POINTS_PER_VOXEL)
    return;
  float *address = voxels + (voxel_index * POINTS_PER_VOXEL + point_id) * 5;
  atomicExch(address + 0, x);
  atomicExch(address + 1, y);
  atomicExch(address + 2, z);
  atomicExch(address + 3, w);
  atomicExch(address + 4, t);
}

cudaError_t generateVoxels_random_launch(
    float *points, unsigned int points_size, float min_x_range, float max_x_range,
    float min_y_range, float max_y_range, float min_z_range, float max_z_range,
    float pillar_x_size, float pillar_y_size, float pillar_z_size,
    int grid_y_size, int grid_x_size, unsigned int *mask, float *voxels,
    cudaStream_t stream) {
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((points_size + threadNum - 1) / threadNum);
  dim3 threads(threadNum);
  generateVoxels_random_kernel<<<blocks, threads, 0, stream>>>(
      points, points_size, min_x_range, max_x_range, min_y_range, max_y_range,
      min_z_range, max_z_range, pillar_x_size, pillar_y_size, pillar_z_size,
      grid_y_size, grid_x_size, mask, voxels);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
                                            int grid_y_size, int grid_x_size,
                                            unsigned int *pillar_num,
                                            float *voxel_features,
                                            unsigned int *voxel_num,
                                            unsigned int *voxel_idxs) {
  unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (voxel_idx >= grid_x_size || voxel_idy >= grid_y_size)
    return;

  unsigned int voxel_index = voxel_idy * grid_x_size + voxel_idx;
  unsigned int count = mask[voxel_index];
  if (!(count > 0))
    return;
  count = count < POINTS_PER_VOXEL ? count : POINTS_PER_VOXEL;

  unsigned int current_pillarId = 0;
  current_pillarId = atomicAdd(pillar_num, 1);

  voxel_num[current_pillarId] = count;

  uint4 idx = {0, 0, voxel_idy, voxel_idx};
  ((uint4 *)voxel_idxs)[current_pillarId] = idx;

  for (int i = 0; i < count; i++) {
    int inIndex = voxel_index * POINTS_PER_VOXEL + i;
    int outIndex = current_pillarId * POINTS_PER_VOXEL + i;
    // ((float4 *)voxel_features)[outIndex] = ((float4 *)voxels)[inIndex];
    voxel_features[outIndex * 5] = voxels[inIndex * 5];
    voxel_features[outIndex * 5 + 1] = voxels[inIndex * 5 + 1];
    voxel_features[outIndex * 5 + 2] = voxels[inIndex * 5 + 2];
    voxel_features[outIndex * 5 + 3] = voxels[inIndex * 5 + 3];
    voxel_features[outIndex * 5 + 4] = voxels[inIndex * 5 + 4];
  }

  // clear buffer for next infer
  atomicExch(mask + voxel_index, 0);
}

// create 4 channels
cudaError_t
generateBaseFeatures_launch(unsigned int *mask, float *voxels, int grid_y_size,
                            int grid_x_size, unsigned int *pillar_num,
                            float *voxel_features, unsigned int *voxel_num,
                            unsigned int *voxel_idxs, cudaStream_t stream) {
  dim3 threads = {32, 32};
  dim3 blocks = {(grid_x_size + threads.x - 1) / threads.x,
                 (grid_y_size + threads.y - 1) / threads.y};

  generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>(
      mask, voxels, grid_y_size, grid_x_size, pillar_num, voxel_features,
      voxel_num, voxel_idxs);
  cudaError_t err = cudaGetLastError();
  return err;
}
} // namespace transfusion