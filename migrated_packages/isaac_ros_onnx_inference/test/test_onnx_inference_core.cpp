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
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif

#include <gtest/gtest.h>

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

using nvidia::isaac_ros::onnx_inference::ExecutionProvider;
using nvidia::isaac_ros::onnx_inference::OnnxInferenceCore;
using nvidia::isaac_ros::onnx_inference::OutputPlacement;
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
  const std::vector<int64_t> shape{1, 3, 640, 640};
  const size_t byte_size = 1U * 3U * 640U * 640U * sizeof(float);
  void * device_input = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, byte_size), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_input, 0, byte_size), cudaSuccess);

  auto input_owner = std::shared_ptr<void>(
    device_input, [](void * pointer) {static_cast<void>(cudaFree(pointer));});
  auto input_buffer = gpu_ros_managed::cuda::adopt_synchronized_external(
    device_input, byte_size, 0, input_owner);
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.emplace_back(
    "images", gpu_ros_managed::TensorDataType::kFloat32, shape, input_buffer);
  auto message = std::make_shared<gpu_ros_managed::ManagedTensorList>(
    std_msgs::msg::Header{}, std::move(tensors));

  std::vector<nvidia::isaac_ros::onnx_inference::OutputTensor> outputs;
  {
    OnnxInferenceCore core(cfg);
    outputs = core.RunInference(
      gpu_ros_managed::ManagedTensorListView(message), OutputPlacement::kDevice);
  }  // The session is deliberately destroyed before the output allocation.
  ASSERT_FALSE(outputs.empty());
  auto * device_buffer =
    std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&outputs.front().storage);
  ASSERT_NE(device_buffer, nullptr);
  ASSERT_TRUE(*device_buffer);
  auto lease = (*device_buffer)->get_blocking_ready_lease();
  ASSERT_NE(lease.data(), nullptr);
  ASSERT_GE(lease.size(), sizeof(float));

  float first_value = 0.0F;
  EXPECT_EQ(
    cudaMemcpy(&first_value, lease.data(), sizeof(first_value), cudaMemcpyDeviceToHost),
    cudaSuccess);
  outputs.clear();
  input_owner.reset();
}
#endif
