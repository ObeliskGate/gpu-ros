// Copyright 2026 Maintainer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#if defined(ORT_CUDA_AVAILABLE) && defined(TEST_CUDA_IO_BINDING_MODEL_PATH)
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

using nvidia::isaac_ros::onnx_inference::ExecutionProvider;
using nvidia::isaac_ros::onnx_inference::DeviceTensorBuffer;
using nvidia::isaac_ros::onnx_inference::OnnxInferenceCore;
using nvidia::isaac_ros::onnx_inference::TensorMemoryKind;
using nvidia::isaac_ros::onnx_inference::TensorView;
using nvidia::isaac_ros::onnx_inference::ParseExecutionProvider;

TEST(OnnxInferenceCoreTest, ParseEpCuda)
{
  EXPECT_EQ(ParseExecutionProvider("cuda"), ExecutionProvider::kCuda);
}

TEST(OnnxInferenceCoreTest, ParseEpCpu)
{
  EXPECT_EQ(ParseExecutionProvider("cpu"), ExecutionProvider::kCpu);
}

TEST(OnnxInferenceCoreTest, ParseEpRocm)
{
  EXPECT_EQ(ParseExecutionProvider("rocm"), ExecutionProvider::kRocm);
}

TEST(OnnxInferenceCoreTest, ParseEpMigraphx)
{
  EXPECT_EQ(ParseExecutionProvider("migraphx"), ExecutionProvider::kMigraphx);
}

TEST(OnnxInferenceCoreTest, ParseEpRocmThrows)
{
#ifndef ORT_ROCM_AVAILABLE
  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = "/dev/null";  // EP check happens before the model is opened
  cfg.ep = ExecutionProvider::kRocm;
  EXPECT_THROW(OnnxInferenceCore core(cfg), std::runtime_error);
#else
  GTEST_SKIP() << "ROCm EP was enabled for this build.";
#endif
}

TEST(OnnxInferenceCoreTest, ParseEpMigraphxThrows)
{
#ifndef ORT_MIGRAPHX_AVAILABLE
  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = "/dev/null";  // EP check happens before the model is opened
  cfg.ep = ExecutionProvider::kMigraphx;
  EXPECT_THROW(OnnxInferenceCore core(cfg), std::runtime_error);
#else
  GTEST_SKIP() << "MIGraphX EP was enabled for this build.";
#endif
}

TEST(OnnxInferenceCoreTest, ParseEpUnknownThrows)
{
  EXPECT_THROW(ParseExecutionProvider("tpu"), std::invalid_argument);
}

#if defined(ORT_CUDA_AVAILABLE) && defined(TEST_CUDA_IO_BINDING_MODEL_PATH)
TEST(OnnxInferenceCoreTest, CudaIoBindingOutputOwnerKeepsDeviceBufferAlive)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  OnnxInferenceCore::Config cfg;
  cfg.model_file_path = TEST_CUDA_IO_BINDING_MODEL_PATH;
  cfg.ep = ExecutionProvider::kCuda;
  cfg.gpu_device_id = 0;
  OnnxInferenceCore core(cfg);

  const std::vector<int64_t> shape{1, 3, 640, 640};
  const size_t byte_size = 1U * 3U * 640U * 640U * sizeof(float);
  void * device_input = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, byte_size), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_input, 0, byte_size), cudaSuccess);

  TensorView input{
    "images", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape,
    device_input, byte_size, TensorMemoryKind::kCudaDevice, 0};

  TensorView short_input = input;
  short_input.byte_size = byte_size - 1;
  EXPECT_THROW(
    core.RunInference({short_input}, TensorMemoryKind::kCudaDevice), std::invalid_argument);

  TensorView negative_shape_input = input;
  negative_shape_input.shape = {1, -1, 640, 640};
  EXPECT_THROW(
    core.RunInference({negative_shape_input}, TensorMemoryKind::kCudaDevice),
    std::invalid_argument);

  auto outputs = core.RunInference({input}, TensorMemoryKind::kCudaDevice);
  ASSERT_FALSE(outputs.empty());
  auto * device_buffer = std::get_if<DeviceTensorBuffer>(&outputs.front().storage);
  ASSERT_NE(device_buffer, nullptr);
  ASSERT_NE(device_buffer->data, nullptr);
  ASSERT_TRUE(device_buffer->owner);
  ASSERT_GE(device_buffer->byte_size, sizeof(float));

  void * output_data = device_buffer->data;
  auto output_owner = std::move(device_buffer->owner);
  outputs.clear();

  float first_value = 0.0F;
  EXPECT_EQ(
    cudaMemcpy(&first_value, output_data, sizeof(first_value), cudaMemcpyDeviceToHost),
    cudaSuccess);
  output_owner.reset();
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}
#endif
