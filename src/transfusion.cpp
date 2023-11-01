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

#include "transfusion/transfusion.hpp"

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
} // namespace

namespace transfusion {
Engine::~Engine() {
  delete (context_);
  delete (engine_);
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

Engine::Engine(std::string plan, cudaStream_t stream) : stream_(stream) {
  std::fstream cache(plan, std::ifstream::in);
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
  if (cache.is_open()) {
    std::cout << "load plan file." << std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    cache.seekg(0, cache.end);
    length = cache.tellg();
    cache.seekg(0, cache.beg);

    data = (char *)malloc(length);
    if (data == NULL) {
      std::cout << "Can't malloc data.\n";
      exit(-1);
    }

    cache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(gLogger_);

    if (runtime == nullptr) {
      std::cout << "load TRT cache0." << std::endl;
      std::cerr << ": runtime null!" << std::endl;
      exit(-1);
    }
    // plugin_ = nvonnxparser::createPluginFactory(gLogger_);
    engine_ = (runtime->deserializeCudaEngine(data, length, 0));
    if (engine_ == nullptr) {
      std::cerr << ": engine null!" << std::endl;
      exit(-1);
    }
    free(data);
    cache.close();
  }

  context_ = engine_->createExecutionContext();
  return;
}

bool Engine::infer() {
  auto status = context_->enqueueV3(stream_);

  return status;
}

bool Engine::setTensorAddress(const char *name, void *data) {
  auto status = context_->setTensorAddress(name, data);
  return status;
}

bool Engine::setInputShape(const char *name, nvinfer1::Dims const &dims) {
  auto status = context_->setInputShape(name, dims);
  return status;
}

Transfusion::Transfusion(std::string plan, cudaStream_t stream)
    : stream_(stream) {
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));

  pre_.reset(new PreprocessCuda(stream_));
  engine_.reset(new Engine(plan, stream_));

  // point cloud to voxels
  voxel_features_size_ =
      MAX_VOXELS * params_.max_num_points_per_pillar * 5 * sizeof(float);
  voxel_num_size_ = MAX_VOXELS * sizeof(unsigned int);
  voxel_idxs_size_ = MAX_VOXELS * 4 * sizeof(unsigned int);

  checkCudaErrors(
      cudaMallocManaged((void **)&voxel_features_, voxel_features_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_num_, voxel_num_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_idxs_, voxel_idxs_size_));

  checkCudaErrors(
      cudaMemsetAsync(voxel_features_, 0, voxel_features_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, stream_));

  // TRT-input
  checkCudaErrors(
      cudaMallocManaged((void **)&params_input_, sizeof(unsigned int)));

  checkCudaErrors(
      cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

  // output of TRT -- input of post-process
  cls_size_ = params_.num_proposals * params_.num_classes * sizeof(float);
  box_size_ = params_.num_proposals * params_.num_box_values * sizeof(float);
  dir_cls_size_ = params_.num_proposals * 2 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&cls_output_, cls_size_));
  checkCudaErrors(cudaMallocManaged((void **)&box_output_, box_size_));
  checkCudaErrors(cudaMallocManaged((void **)&dir_cls_output_, dir_cls_size_));

  res_.reserve(100);
  return;
}

Transfusion::~Transfusion(void) {
  pre_.reset();
  engine_.reset();

  checkCudaErrors(cudaFree(voxel_features_));
  checkCudaErrors(cudaFree(voxel_num_));
  checkCudaErrors(cudaFree(voxel_idxs_));

  checkCudaErrors(cudaFree(params_input_));

  checkCudaErrors(cudaFree(cls_output_));
  checkCudaErrors(cudaFree(box_output_));
  checkCudaErrors(cudaFree(dir_cls_output_));

  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

int Transfusion::infer(void *points_data, unsigned int points_size,
                       std::vector<Bndbox> &pred) {
  float generateVoxelsTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));

  pre_->generateVoxels((float *)points_data, points_size, params_input_,
                       voxel_features_, voxel_num_, voxel_idxs_);

  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&generateVoxelsTime, start_, stop_));
  unsigned int params_input_cpu;
  checkCudaErrors(cudaMemcpy(&params_input_cpu, params_input_,
                             sizeof(unsigned int), cudaMemcpyDefault));
  std::cout << "find pillar_num: " << params_input_cpu << std::endl;

  float inferTime = 0.0f;

  engine_->setTensorAddress("voxels", voxel_features_);
  engine_->setInputShape(
      "voxels", nvinfer1::Dims3{static_cast<int32_t>(params_input_cpu), 20, 5});
  engine_->setTensorAddress("num_points", voxel_num_);
  nvinfer1::Dims num_points_dim;
  num_points_dim.nbDims = 1;
  num_points_dim.d[0] = static_cast<int32_t>(params_input_cpu);
  engine_->setInputShape("num_points", num_points_dim);
  engine_->setTensorAddress("coors", voxel_idxs_);
  engine_->setInputShape(
      "coors", nvinfer1::Dims2{static_cast<int32_t>(params_input_cpu), 4});
  engine_->setTensorAddress("cls_score0", cls_output_);
  engine_->setTensorAddress("bbox_pred0", box_output_);
  engine_->setTensorAddress("dir_cls_pred0", dir_cls_output_);
  for (int i = 0; i < 5; ++i) engine_->infer();
  checkCudaErrors(cudaEventRecord(start_, stream_));
  for (int i = 0; i < 10; ++i) engine_->infer();

  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&inferTime, start_, stop_));

  for (size_t i = 0; i < params_.num_proposals; ++i) {
    auto class_id = 0;
    auto max_score = cls_output_[i];
    for (size_t j = 1; j < params_.num_classes; ++j) {
      auto score = cls_output_[params_.num_proposals * j + i];
      if (max_score < score) {
        max_score = score;
        class_id = j;
      }
    }
    if (max_score < params_.score_thresh) {
      continue;
    }
    pred.emplace_back(transfusion::Bndbox{
        box_output_[i] * params_.out_size_factor * params_.pillar_x_size +
            params_.min_x_range,
        box_output_[i + params_.num_proposals] * params_.out_size_factor *
                params_.pillar_y_size +
            params_.min_y_range,
        box_output_[i + 2 * params_.num_proposals],
        std::exp(box_output_[i + 3 * params_.num_proposals]),
        std::exp(box_output_[i + 4 * params_.num_proposals]),
        std::exp(box_output_[i + 5 * params_.num_proposals]),
        std::atan2(dir_cls_output_[i], dir_cls_output_[i + params_.num_proposals]),
        class_id,
        max_score,
    });
  }
  std::cout << "TIME: generateVoxels: " << generateVoxelsTime << " ms."
            << std::endl;
  std::cout << "TIME: infer: " << inferTime / 10.0 << " ms." << std::endl;
  return 0;
}
} // namespace transfusion