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

#include <gtest/gtest.h>

#include <stdexcept>

#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

using nvidia::isaac_ros::onnx_inference::ExecutionProvider;
using nvidia::isaac_ros::onnx_inference::OnnxInferenceCore;
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
