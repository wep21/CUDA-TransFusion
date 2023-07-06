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

#ifndef TRANSFUSION__KERNELS_HPP_
#define TRANSFUSION__KERNELS_HPP_

#include "transfusion/parameters.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

#define checkCudaErrors(status)                                                \
  {                                                                            \
    if (status != 0) {                                                         \
      std::cout << "Cuda failure: " << cudaGetErrorString(status)              \
                << " at line " << __LINE__ << " in file " << __FILE__          \
                << " error status: " << status << std::endl;                   \
      abort();                                                                 \
    }                                                                          \
  }

namespace transfusion {
static constexpr int THREADS_FOR_VOXEL = 256; // threads number for a block
static constexpr int POINTS_PER_VOXEL = 20;   // depands on "params.h"
static constexpr int WARP_SIZE = 32;      // one warp(32 threads) for one pillar
static constexpr int WARPS_PER_BLOCK = 4; // four warp for one block
static constexpr int FEATURES_SIZE =
    10; // features maps number depands on "params.h"
// one thread deals with one pillar and a block has PILLARS_PER_BLOCK threads
static constexpr int PILLARS_PER_BLOCK = 64;
// feature count for one pillar depands on "params.h"
static constexpr int PILLAR_FEATURE_SIZE = 64;
static constexpr int MAX_VOXELS = 40000;

cudaError_t generateVoxels_random_launch(
    float *points, unsigned int points_size, float min_x_range, float max_x_range,
    float min_y_range, float max_y_range, float min_z_range, float max_z_range,
    float pillar_x_size, float pillar_y_size, float pillar_z_size,
    int grid_y_size, int grid_x_size, unsigned int *mask, float *voxels,
    cudaStream_t stream = 0);

cudaError_t
generateBaseFeatures_launch(unsigned int *mask, float *voxels, int grid_y_size,
                            int grid_x_size, unsigned int *pillar_num,
                            float *voxel_features, unsigned int *voxel_num,
                            unsigned int *voxel_idxs, cudaStream_t stream = 0);

} // namespace transfusion

#endif // TRANSFUSION__KERNELS_HPP_