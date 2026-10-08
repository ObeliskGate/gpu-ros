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

#if defined(ORT_CUDA_AVAILABLE) && defined(TEST_CUDA_IO_BINDING_MODEL_PATH)
#include <cuda_runtime.h>
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif

#include <gtest/gtest.h>

#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"

#ifdef GPU_ROS_ORT_CANCELLATION_TEST
#include "gpu_ros_managed_hip/hip_backend.hpp"
#include "onnx_inference_cancellation_test_peer.hpp"
#endif

using gpu_ros::onnx_inference::ExecutionProvider;
using gpu_ros::onnx_inference::OnnxInferenceCore;
using gpu_ros::onnx_inference::OutputPlacement;
using gpu_ros::onnx_inference::ParseExecutionProvider;

TEST(TensorContractTest, ParsesConcreteFloatAndInt64Contracts)
{
  const auto contracts = gpu_ros::onnx_inference::ParseTensorContracts(
    {"images=float32[1,3,640,640]", "orig_target_sizes=int64[1,2]"}, "managed_input_contracts");
  ASSERT_EQ(contracts.size(), 2U);
  EXPECT_EQ(contracts[0].name, "images");
  EXPECT_EQ(contracts[0].dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(contracts[0].shape, (std::vector<int64_t>{1, 3, 640, 640}));
  EXPECT_EQ(gpu_ros::onnx_inference::TensorByteSize(contracts[1]), 16U);
}

TEST(TensorContractTest, RejectsDynamicAndDuplicateSpecifications)
{
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"images=float32[1,-1,640,640]"}, "managed_input_contracts"),
    std::invalid_argument);
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"images=float32[1]", "images=int64[1]"}, "managed_input_contracts"),
    std::invalid_argument);
  EXPECT_THROW(gpu_ros::onnx_inference::ParseTensorContracts(
                 {"=float32[1]"}, "managed_input_contracts"),
    std::invalid_argument);
}

TEST(TensorContractTest, RejectsOverflowingByteSizes)
{
  const auto contracts = gpu_ros::onnx_inference::ParseTensorContracts(
    {"large=float32[9223372036854775807,2]"}, "managed_output_contracts");
  EXPECT_THROW(
    gpu_ros::onnx_inference::TensorByteSize(contracts.front()), std::overflow_error);
}

TEST(TensorContractTest, ErrorsUseCallerParameterName)
{
  try {
    static_cast<void>(gpu_ros::onnx_inference::ParseTensorContracts(
      {"output=float16[1]"}, "output_contracts"));
    FAIL() << "Unsupported dtype was accepted";
  } catch (const std::invalid_argument & error) {
    EXPECT_NE(std::string(error.what()).find("output_contracts"), std::string::npos);
    EXPECT_EQ(std::string(error.what()).find("managed_output_contracts"), std::string::npos);
  }
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

#if defined(ORT_CUDA_AVAILABLE) && defined(TEST_CUDA_IO_BINDING_MODEL_PATH)
TEST(OnnxInferenceCoreTest, CudaIoBindingOutputOwnerKeepsDeviceBufferAlive)
{
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0) << "CUDA binding acceptance requires a real device";

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
    device_input, [](void * pointer) { static_cast<void>(cudaFree(pointer)); });
  auto input_buffer =
    gpu_ros_managed::cuda::adopt_synchronized_external(device_input, byte_size, 0, input_owner);
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.emplace_back("images", gpu_ros_managed::TensorDataType::kFloat32, shape, input_buffer);
  auto message = std::make_shared<gpu_ros_managed::ManagedTensorBundle>(
    std_msgs::msg::Header{}, std::move(tensors));

  std::vector<gpu_ros::onnx_inference::OutputTensor> outputs;
  {
    OnnxInferenceCore core(cfg);
    outputs = core.RunInference(
      gpu_ros_managed::ManagedTensorBundleView(message), OutputPlacement::kDevice);
  } // The core is gone; ORT-owned output storage must retain its session.
  ASSERT_FALSE(outputs.empty());
  auto * device_buffer =
    std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&outputs.front().storage);
  ASSERT_NE(device_buffer, nullptr);
  ASSERT_TRUE(*device_buffer);
  auto lease = (*device_buffer)->get_blocking_ready_lease();
  ASSERT_NE(lease.data(), nullptr);
  ASSERT_GE(lease.size(), sizeof(float));

  float first_value = 0.0F;
  EXPECT_EQ(cudaMemcpy(&first_value, lease.data(), sizeof(first_value), cudaMemcpyDeviceToHost),
    cudaSuccess);
  outputs.clear();
  input_owner.reset();
}
#endif

#ifdef GPU_ROS_ORT_CANCELLATION_TEST
namespace
{

using CancellationPeer = gpu_ros::onnx_inference::OnnxInferenceCancellationTestPeer;
using CancellationPoint = CancellationPeer::Point;
using OutputTensor = gpu_ros::onnx_inference::OutputTensor;
using namespace std::chrono_literals;

class ManagedHipCancellationTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "MIGraphXExecutionProvider") ==
        providers.end())
    {
      GTEST_SKIP() << "The loaded ONNX Runtime has no MIGraphX provider.";
    }
    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if (status == hipErrorNoDevice || (status == hipSuccess && count == 0)) {
      GTEST_SKIP() << "No HIP device is available.";
    }
    ASSERT_EQ(status, hipSuccess);
    CancellationPeer::Arm(CancellationPoint::kNone);

    const std::vector<float> pixels(12, 0.25F);
    const std::vector<int64_t> sizes{2, 2};
    auto images = gpu_ros_managed::hip::allocate(pixels.size() * sizeof(float), 0);
    images->copy_from_host_blocking(pixels.data(), pixels.size() * sizeof(float));
    auto target_sizes = gpu_ros_managed::hip::allocate(sizes.size() * sizeof(int64_t), 0);
    target_sizes->copy_from_host_blocking(sizes.data(), sizes.size() * sizeof(int64_t));
    std::vector<gpu_ros_managed::ManagedTensor> tensors;
    tensors.emplace_back("images", gpu_ros_managed::TensorDataType::kFloat32,
      std::vector<int64_t>{1, 3, 2, 2}, images);
    tensors.emplace_back("orig_target_sizes", gpu_ros_managed::TensorDataType::kInt64,
      std::vector<int64_t>{1, 2}, target_sizes);
    inputs_ = std::make_shared<gpu_ros_managed::ManagedTensorBundle>(
      std_msgs::msg::Header{}, std::move(tensors));
  }

  void TearDown() override
  {
    CancellationPeer::Arm(CancellationPoint::kNone);
    inputs_.reset();
    EXPECT_TRUE(gpu_ros_managed::hip::wait_for_pending_releases(5s));
  }

  static OnnxInferenceCore::Config Config(bool strict)
  {
    OnnxInferenceCore::Config cfg;
    cfg.model_file_path = TEST_HIP_CANCELLATION_MODEL_PATH;
    cfg.ep = ExecutionProvider::kMigraphx;
    cfg.transport = "managed";
    cfg.managed_pool_capacity = 1;
    cfg.managed_pool_wait_timeout = 0ms;
    if (strict) {
      cfg.managed_io_contract = "hip_managed_strict";
      cfg.managed_input_contracts = {"images=float32[1,3,2,2]", "orig_target_sizes=int64[1,2]"};
      cfg.managed_output_contracts = {
        "labels=int64[1,300]", "boxes=float32[1,300,4]", "scores=float32[1,300]"};
    }
    return cfg;
  }

  std::vector<OutputTensor> Run(OnnxInferenceCore & core)
  {
    return core.RunInference(
      gpu_ros_managed::ManagedTensorBundleView(inputs_), OutputPlacement::kDevice);
  }

  static void ExpectModelOutputs(const std::vector<OutputTensor> & outputs)
  {
    ASSERT_EQ(outputs.size(), 3U);
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto & output = outputs[i];
      const auto * storage =
        std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&output.storage);
      ASSERT_NE(storage, nullptr);
      ASSERT_TRUE(*storage);
      EXPECT_FALSE((*storage)->failed());
      if (i == 0) {
        EXPECT_EQ(output.name, "labels");
        EXPECT_EQ(output.dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        EXPECT_EQ(output.shape, (std::vector<int64_t>{1, 300}));
        std::vector<int64_t> actual(300);
        (*storage)->copy_to_host_blocking(actual.data(), actual.size() * sizeof(int64_t));
        std::vector<int64_t> expected(300, 0);
        expected[0] = 7;
        EXPECT_EQ(actual, expected);
      } else {
        EXPECT_EQ(output.name, i == 1 ? "boxes" : "scores");
        EXPECT_EQ(output.dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        EXPECT_EQ(output.shape,
          i == 1 ? (std::vector<int64_t>{1, 300, 4}) : (std::vector<int64_t>{1, 300}));
        std::vector<float> actual(i == 1 ? 1200 : 300);
        (*storage)->copy_to_host_blocking(actual.data(), actual.size() * sizeof(float));
        std::vector<float> expected(actual.size(), 0.0F);
        if (i == 1) {
          expected[0] = 10.0F;
          expected[1] = 20.0F;
          expected[2] = 30.0F;
          expected[3] = 50.0F;
        } else {
          expected[0] = 0.95F;
        }
        for (size_t index = 0; index < actual.size(); ++index) {
          EXPECT_FLOAT_EQ(actual[index], expected[index]) << output.name << "[" << index << "]";
        }
      }
    }
  }

  std::shared_ptr<gpu_ros_managed::ManagedTensorBundle> inputs_;
};

class ManagedHipPreSubmissionTest : public ManagedHipCancellationTest,
                                    public ::testing::WithParamInterface<CancellationPoint>
{
};

TEST_P(ManagedHipPreSubmissionTest, StrictCancellationRestoresEveryPoolAndDrains)
{
  OnnxInferenceCore core(Config(true));
  CancellationPeer::Arm(GetParam());
  EXPECT_THROW(Run(core), std::runtime_error);
  ASSERT_TRUE(CancellationPeer::fired);
  ASSERT_TRUE(CancellationPeer::first_buffer);
  EXPECT_FALSE(CancellationPeer::first_buffer->failed());
  EXPECT_EQ(
    CancellationPeer::first_buffer->readiness(), gpu_ros_managed::BufferReadiness::kNotReady);
  if (GetParam() == CancellationPoint::kSecondBind) {
    ASSERT_TRUE(CancellationPeer::second_buffer);
    EXPECT_FALSE(CancellationPeer::second_buffer->failed());
  }
  std::weak_ptr<gpu_ros_managed::DeviceBuffer> released = CancellationPeer::first_buffer;
  CancellationPeer::first_buffer.reset();
  CancellationPeer::second_buffer.reset();
  EXPECT_TRUE(released.expired());
  EXPECT_EQ(CancellationPeer::Available(core), (std::vector<size_t>{1, 1, 1}));
  EXPECT_TRUE(core.shutdown(5s));
}

TEST_P(ManagedHipPreSubmissionTest, CompatibilityCancelsAndRunsNumericFallback)
{
  OnnxInferenceCore core(Config(false));
  CancellationPeer::Arm(GetParam());
  auto outputs = Run(core);
  ASSERT_TRUE(CancellationPeer::fired);
  ExpectModelOutputs(outputs);
  ASSERT_TRUE(CancellationPeer::first_buffer);
  EXPECT_FALSE(CancellationPeer::first_buffer->failed());
  EXPECT_EQ(
    CancellationPeer::first_buffer->readiness(), gpu_ros_managed::BufferReadiness::kNotReady);
  EXPECT_TRUE(CancellationPeer::retained_owner.expired());
  if (GetParam() == CancellationPoint::kSecondBind) {
    ASSERT_TRUE(CancellationPeer::second_buffer);
    EXPECT_FALSE(CancellationPeer::second_buffer->failed());
    EXPECT_EQ(
      CancellationPeer::second_buffer->readiness(), gpu_ros_managed::BufferReadiness::kNotReady);
    EXPECT_TRUE(CancellationPeer::second_retained_owner.expired());
  }
  // Fresh really is reusable, not merely a non-failed wrapper.
  {
    auto writer = CancellationPeer::first_buffer->get_synchronized_write_handle();
    writer.cancel();
  }
  std::weak_ptr<gpu_ros_managed::DeviceBuffer> released = CancellationPeer::first_buffer;
  CancellationPeer::first_buffer.reset();
  CancellationPeer::second_buffer.reset();
  EXPECT_TRUE(released.expired());
  outputs.clear();
  EXPECT_TRUE(gpu_ros_managed::hip::wait_for_pending_releases(5s));
}

INSTANTIATE_TEST_SUITE_P(BeforeRun, ManagedHipPreSubmissionTest,
  ::testing::Values(CancellationPoint::kFirstWriter, CancellationPoint::kSecondBind));

class ManagedHipSubmissionTest : public ManagedHipCancellationTest,
                                 public ::testing::WithParamInterface<bool>
{
};

TEST_P(ManagedHipSubmissionTest, PossibleSubmissionFailsInsteadOfCancelling)
{
  const bool strict = GetParam();
  OnnxInferenceCore core(Config(strict));
  CancellationPeer::Arm(CancellationPoint::kSubmissionBoundary);
  EXPECT_THROW(Run(core), std::runtime_error);
  ASSERT_TRUE(CancellationPeer::fired);
  ASSERT_TRUE(CancellationPeer::first_buffer);
  EXPECT_TRUE(CancellationPeer::first_buffer->failed());
  EXPECT_THROW(CancellationPeer::first_buffer->get_synchronized_write_handle(), std::logic_error);
  ASSERT_TRUE(CancellationPeer::second_buffer);
  EXPECT_TRUE(CancellationPeer::second_buffer->failed());
  CancellationPeer::first_buffer.reset();
  CancellationPeer::second_buffer.reset();
  EXPECT_TRUE(gpu_ros_managed::hip::wait_for_pending_releases(5s));
  if (strict) {
    EXPECT_FALSE(core.healthy());
    EXPECT_EQ(CancellationPeer::Available(core), (std::vector<size_t>{0, 0, 0}));
    EXPECT_FALSE(core.shutdown(0ms));
  } else {
    // The wrapper can disappear, but uncertain work must keep its source owner
    // and allocation orphaned rather than taking the pre-submission rollback.
    EXPECT_FALSE(CancellationPeer::retained_owner.expired());
    EXPECT_FALSE(CancellationPeer::second_retained_owner.expired());
  }
}

TEST_P(ManagedHipSubmissionTest, UninjectedRunSynchronizesAndReleasesOutputs)
{
  const bool strict = GetParam();
  OnnxInferenceCore core(Config(strict));
  auto outputs = Run(core);
  EXPECT_FALSE(CancellationPeer::fired);
  ExpectModelOutputs(outputs);
  ASSERT_TRUE(CancellationPeer::first_buffer);
  EXPECT_FALSE(CancellationPeer::first_buffer->failed());
  outputs.clear();
  CancellationPeer::first_buffer.reset();
  CancellationPeer::second_buffer.reset();
  EXPECT_TRUE(gpu_ros_managed::hip::wait_for_pending_releases(5s));
  if (strict) {
    EXPECT_EQ(CancellationPeer::Available(core), (std::vector<size_t>{1, 1, 1}));
    EXPECT_TRUE(core.shutdown(5s));
  } else {
    EXPECT_TRUE(CancellationPeer::retained_owner.expired());
    EXPECT_TRUE(CancellationPeer::second_retained_owner.expired());
  }
}

INSTANTIATE_TEST_SUITE_P(StrictAndCompatibility, ManagedHipSubmissionTest, ::testing::Bool());

} // namespace
#else
TEST(OnnxInferenceCoreTest, ManagedHipCancellationRequiresMigraphx)
{
  GTEST_SKIP() << "Real managed output cancellation tests require ORT_ENABLE_MIGRAPHX=ON "
                  "and the HIP backend.";
}
#endif
