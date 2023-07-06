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

#ifndef TRANSFUSION__TRANSFUSION_HPP_
#define TRANSFUSION__TRANSFUSION_HPP_

#include <iostream>
#include <memory>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include "transfusion/preprocess.hpp"

namespace transfusion {
// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR ||
        severity == Severity::kINFO) {
      std::cerr << "tensorrt: " << msg << std::endl;
    }
  }
};

class Engine {
private:
  Parameters params_;

  cudaEvent_t start_, stop_;

  Logger gLogger_;
  nvinfer1::IExecutionContext *context_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;

  cudaStream_t stream_ = 0;

public:
  Engine(std::string plan, cudaStream_t stream = 0);
  ~Engine();

  bool infer();
  bool setTensorAddress(const char * name, void * data);
  bool setInputShape(const char * name, nvinfer1::Dims const & dims);
};

struct Bndbox {
  float x;
  float y;
  float z;
  float w;
  float l;
  float h;
  float rt;
  int id;
  float score;
  Bndbox(){};
  Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_,
         int id_, float score_)
      : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_),
        score(score_) {}
};

class Transfusion {
private:
  Parameters params_;

  cudaEvent_t start_, stop_;
  cudaStream_t stream_;

  std::shared_ptr<PreprocessCuda> pre_;
  std::shared_ptr<Engine> engine_;

  // input of pre-process
  float *voxel_features_ = nullptr;
  unsigned int *voxel_num_ = nullptr;
  unsigned int *voxel_idxs_ = nullptr;
  unsigned int *pillar_num_ = nullptr;

  unsigned int voxel_features_size_ = 0;
  unsigned int voxel_num_size_ = 0;
  unsigned int voxel_idxs_size_ = 0;

  // TRT-input
  unsigned int *params_input_ = nullptr;

  // output of TRT -- input of post-process
  float *cls_output_ = nullptr;
  float *box_output_ = nullptr;
  float *dir_cls_output_ = nullptr;
  unsigned int cls_size_;
  unsigned int box_size_;
  unsigned int dir_cls_size_;

  std::vector<Bndbox> res_;

public:
  Transfusion(std::string plan, cudaStream_t stream = 0);
  ~Transfusion();
  int infer(void *points, unsigned int point_size, std::vector<Bndbox> &res);
};
} // namespace transfusion

#endif // TRANSFUSION__TRANSFUSION_HPP_
