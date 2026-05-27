// Copyright 2026 - Apache-2.0
// Task 1: compile-time smoke test for OnnxInferenceCore.
// Task 2 will add session-load and inference tests.
#include <gtest/gtest.h>
#include <stdexcept>
#include "isaac_ros_onnx_inference/onnx_inference_core.hpp"

using nvidia::isaac_ros::onnx_inference::ExecutionProvider;
using nvidia::isaac_ros::onnx_inference::ParseExecutionProvider;

TEST(OnnxInferenceCoreTest, ParseEpCuda)
{
  EXPECT_EQ(ParseExecutionProvider("cuda"), ExecutionProvider::kCuda);
}

TEST(OnnxInferenceCoreTest, ParseEpCpu)
{
  EXPECT_EQ(ParseExecutionProvider("cpu"), ExecutionProvider::kCpu);
}

TEST(OnnxInferenceCoreTest, ParseEpRocmThrows)
{
  // ROCm EP is not built in Phase 1; requesting it must throw, not silently degrade.
  EXPECT_THROW(
    {
      nvidia::isaac_ros::onnx_inference::OnnxInferenceCore::Config cfg;
      cfg.model_file_path = "/dev/null";  // won't be opened; EP check happens first
      cfg.ep = ExecutionProvider::kRocm;
      nvidia::isaac_ros::onnx_inference::OnnxInferenceCore core(cfg);
    },
    std::runtime_error);
}

TEST(OnnxInferenceCoreTest, ParseEpUnknownThrows)
{
  EXPECT_THROW(ParseExecutionProvider("tpu"), std::invalid_argument);
}
