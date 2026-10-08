// Copyright 2026 Boshen Chen
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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>


#include <gtest/gtest.h>

#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"


using gpu_ros::onnx_inference::ExecutionProvider;
using gpu_ros::onnx_inference::OnnxInferenceCore;
using gpu_ros::onnx_inference::OutputPlacement;
using gpu_ros::onnx_inference::ParseExecutionProvider;

TEST(TensorContractTest, ParsesConcreteFloatAndInt64Contracts)
{
  const auto contracts = gpu_ros::onnx_inference::ParseTensorContracts(
    {"images=float32[1,3,640,640]", "orig_target_sizes=int64[1,2]"}, "input_contracts");
  ASSERT_EQ(contracts.size(), 2U);
  EXPECT_EQ(contracts[0].name, "images");
  EXPECT_EQ(contracts[0].dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(contracts[0].shape, (std::vector<int64_t>{1, 3, 640, 640}));
  EXPECT_EQ(gpu_ros::onnx_inference::TensorByteSize(contracts[1]), 16U);
}

TEST(TensorContractTest, RejectsDynamicAndDuplicateSpecifications)
{
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"images=float32[1,-1,640,640]"}, "input_contracts"),
    std::invalid_argument);
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"images=float32[1]", "images=int64[1]"}, "input_contracts"),
    std::invalid_argument);
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"=float32[1]"}, "input_contracts"),
    std::invalid_argument);
}

TEST(TensorContractTest, RejectsOverflowingByteSizes)
{
  const auto contracts = gpu_ros::onnx_inference::ParseTensorContracts(
    {"large=float32[9223372036854775807,2]"}, "output_contracts");
  EXPECT_THROW(
    gpu_ros::onnx_inference::TensorByteSize(contracts.front()), std::overflow_error);
}


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
  cfg.model_file_path = "/dev/null"; // EP check happens before the model is opened
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
  cfg.model_file_path = "/dev/null"; // EP check happens before the model is opened
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

