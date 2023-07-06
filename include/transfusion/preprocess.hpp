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

#ifndef TRANSFUSION__PREPROCESS_HPP_
#define TRANSFUSION__PREPROCESS_HPP_

#include "transfusion/kernels.hpp"

namespace transfusion {
class PreprocessCuda {
private:
  Parameters params_;
  unsigned int *mask_;
  float *voxels_;
  int *voxelsList_;
  float *params_cuda_;
  cudaStream_t stream_ = 0;

public:
  PreprocessCuda(cudaStream_t stream_ = 0);
  ~PreprocessCuda();

  // points cloud -> voxels (BEV) -> feature*4
  int generateVoxels(float *points, unsigned int points_size,
                     unsigned int *pillar_num, float *voxel_features,
                     unsigned int *voxel_num, unsigned int *voxel_idxs);

};
} // namespace transfusion

#endif  // TRANSFUSION__PREPROCESS_HPP_