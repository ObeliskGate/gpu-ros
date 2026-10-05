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
//
#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace inference = gpu_ros::onnx_inference;
using namespace gpu_ros_managed;
namespace
{
struct Transactions
{
  size_t cancelled{0};
  size_t failed{0};
  size_t completed{0};
  bool throw_before_submit{false};
  bool throw_after_submit{false};
  std::vector<void *> pointers;
  std::weak_ptr<const DeviceBufferAttachment> attachment;
  std::weak_ptr<const void> input_owner;
};
class ObservedBatch final : public inference::DeviceOutputBatch
{
public:
  ObservedBatch(std::unique_ptr<inference::DeviceOutputBatch> batch, Transactions & observations)
      : batch_(std::move(batch)), observations_(observations)
  {
  }
  const std::vector<std::shared_ptr<DeviceBuffer>> & buffers() const override
  {
    return batch_->buffers();
  }
  void * pointer(size_t index) const override
  {
    if (observations_.throw_before_submit) {
      throw std::runtime_error("binding setup failure");
    }
    return batch_->pointer(index);
  }
  void RetainOwner(std::shared_ptr<const void> owner) override
  {
    observations_.input_owner = owner;
    batch_->RetainOwner(std::move(owner));
  }
  void CompleteAfterSync() override
  {
    if (observations_.throw_after_submit) {
      throw std::runtime_error("completion failure");
    }
    batch_->CompleteAfterSync();
    ++observations_.completed;
  }
  void CancelBeforeSubmit() noexcept override
  {
    batch_->CancelBeforeSubmit();
    ++observations_.cancelled;
  }
  void FailAfterSubmit() noexcept override
  {
    batch_->FailAfterSubmit();
    ++observations_.failed;
  }
  void CopyFrom(const std::vector<inference::OutputTensor> & tensors) override
  {
    batch_->CopyFrom(tensors);
  }

private:
  std::unique_ptr<inference::DeviceOutputBatch> batch_;
  Transactions & observations_;
};
class ObservedAllocator final : public inference::DeviceOutputAllocator
{
public:
  ObservedAllocator(inference::DeviceOutputAllocator & allocator, Transactions & observations)
      : allocator_(allocator), observations_(observations)
  {
  }
  std::unique_ptr<inference::DeviceOutputBatch> Allocate(const std_msgs::msg::Header & header,
    const std::vector<inference::DeviceOutputSpec> & specs) override
  {
    auto batch = allocator_.Allocate(header, specs);
    for (size_t i = 0; i < specs.size(); ++i) {
      observations_.pointers.push_back(batch->pointer(i));
    }
    if (!batch->buffers().empty()) {
      observations_.attachment = batch->buffers()[0]->attachment();
    }
    return std::make_unique<ObservedBatch>(std::move(batch), observations_);
  }

private:
  inference::DeviceOutputAllocator & allocator_;
  Transactions & observations_;
};
class NativeOutputTest : public ::testing::TestWithParam<const char *>
{
protected:
  void SetUp() override
  {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    node = std::make_shared<rclcpp::Node>("native_output_test");
    node->declare_parameter<std::string>("execution_provider", "cuda");
    node->declare_parameter<int>("gpu_device_id", 0);
    io = inference::CreateTensorBundleIO(node.get(), GetParam(), false);
    ASSERT_NE(io->device_output_allocator(), nullptr);
  }
  void TearDown() override
  {
    io.reset();
    node.reset();
    EXPECT_TRUE(cuda::wait_for_pending_releases(std::chrono::seconds(5)));
    rclcpp::shutdown();
  }
  inference::OnnxInferenceCore::Config Config(bool contract = true)
  {
    inference::OnnxInferenceCore::Config cfg;
    cfg.model_file_path = TEST_DYNAMIC_OUTPUT_MODEL_PATH;
    cfg.ep = inference::ExecutionProvider::kCuda;
    cfg.transport = GetParam();
    if (contract) {
      cfg.managed_output_contracts = {"result=float32[1,4]"};
    }
    return cfg;
  }
  std::shared_ptr<ManagedTensorBundle> Input()
  {
    const float values[]{2, 4, 6, 8};
    auto buffer = cuda::allocate(sizeof(values), 0);
    buffer->copy_from_host_blocking(values, sizeof(values));
    std_msgs::msg::Header header;
    header.frame_id = "per_frame";
    header.stamp.sec = 17;
    return std::make_shared<ManagedTensorBundle>(
      header, std::vector<ManagedTensor>{
                ManagedTensor("values", TensorDataType::kFloat32, {1, 4}, buffer)});
  }
  std::shared_ptr<rclcpp::Node> node;
  std::unique_ptr<inference::ITensorBundleIO> io;
};
TEST_P(NativeOutputTest, DirectPointersAndPayloadSurviveSessionAndNode)
{
  Transactions observations;
  ObservedAllocator allocator(*io->device_output_allocator(), observations);
  std::vector<inference::OutputTensor> outputs;
  {
    inference::OnnxInferenceCore core(Config());
    outputs = core.RunInference(
      ManagedTensorBundleView(Input()), inference::OutputPlacement::kDevice, nullptr, &allocator);
  }
  io.reset();
  node.reset();
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 4}));
  auto buffer = std::get<std::shared_ptr<DeviceBuffer>>(outputs[0].storage);
  ASSERT_TRUE(buffer->attachment());
  EXPECT_EQ(buffer->get_blocking_ready_lease().data(), observations.pointers[0]);
  EXPECT_EQ(observations.completed, 1U);
  EXPECT_TRUE(observations.input_owner.expired());
  float values[4]{};
  ASSERT_EQ(cudaMemcpy(values, observations.pointers[0], sizeof(values), cudaMemcpyDeviceToHost),
    cudaSuccess);
  EXPECT_EQ((std::vector<float>(values, values + 4)), (std::vector<float>{3, 5, 7, 9}));
}
TEST_P(NativeOutputTest, DynamicFallbackCopiesActualShapeAndContent)
{
  inference::OnnxInferenceCore core(Config(false));
  auto outputs = core.RunInference(ManagedTensorBundleView(Input()),
    inference::OutputPlacement::kDevice, nullptr, io->device_output_allocator());
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 4}));
  auto buffer = std::get<std::shared_ptr<DeviceBuffer>>(outputs[0].storage);
  ASSERT_TRUE(buffer->attachment());
  float values[4]{};
  ASSERT_EQ(cudaMemcpy(values, buffer->get_blocking_ready_lease().data(), sizeof(values),
              cudaMemcpyDeviceToHost),
    cudaSuccess);
  EXPECT_EQ((std::vector<float>(values, values + 4)), (std::vector<float>{3, 5, 7, 9}));
  EXPECT_NE(core.OutputBindingProbeReport().find("explicitly copied D2D"), std::string::npos);
}
TEST_P(NativeOutputTest, SetupFailureCancelsAndSubmittedFailureRetainsNativeStorage)
{
  Transactions cancelled;
  cancelled.throw_before_submit = true;
  ObservedAllocator cancel_allocator(*io->device_output_allocator(), cancelled);
  inference::OnnxInferenceCore core(Config());
  EXPECT_THROW(core.RunInference(ManagedTensorBundleView(Input()),
                 inference::OutputPlacement::kDevice, nullptr, &cancel_allocator),
    std::runtime_error);
  EXPECT_EQ(cancelled.cancelled, 1U);
  EXPECT_EQ(cancelled.failed, 0U);
  EXPECT_TRUE(cancelled.attachment.expired());
  EXPECT_TRUE(cancelled.input_owner.expired());
  Transactions failed;
  failed.throw_after_submit = true;
  ObservedAllocator fail_allocator(*io->device_output_allocator(), failed);
  EXPECT_THROW(core.RunInference(ManagedTensorBundleView(Input()),
                 inference::OutputPlacement::kDevice, nullptr, &fail_allocator),
    std::runtime_error);
  EXPECT_EQ(failed.failed, 1U);
  EXPECT_EQ(failed.cancelled, 0U);
  EXPECT_FALSE(failed.attachment.expired());
  EXPECT_FALSE(failed.input_owner.expired());
}
TEST_P(NativeOutputTest, ContractMismatchIsRejectedRatherThanFallback)
{
  auto cfg = Config();
  cfg.managed_output_contracts = {"result=int64[1,4]"};
  EXPECT_THROW(inference::OnnxInferenceCore core(cfg), std::invalid_argument);
  cfg = Config();
  cfg.managed_output_contracts = {"missing=float32[1,4]"};
  EXPECT_THROW(inference::OnnxInferenceCore core(cfg), std::invalid_argument);
  cfg = Config();
  cfg.managed_output_contracts = {"result=float32[1,5]"};
  inference::OnnxInferenceCore core(cfg);
  EXPECT_ANY_THROW(core.RunInference(ManagedTensorBundleView(Input()),
    inference::OutputPlacement::kDevice, nullptr, io->device_output_allocator()));
}
TEST_P(NativeOutputTest, RealYoloUsesFormalDirectOutputContract)
{
  auto cfg = Config();
  cfg.model_file_path = TEST_CUDA_IO_BINDING_MODEL_PATH;
  cfg.managed_output_contracts = {"output0=float32[1,84,8400]"};
  const size_t bytes = 1U * 3U * 640U * 640U * sizeof(float);
  auto buffer = cuda::allocate(bytes, 0);
  std::vector<float> zeros(bytes / sizeof(float), 0);
  buffer->copy_from_host_blocking(zeros.data(), bytes);
  auto input = std::make_shared<ManagedTensorBundle>(std_msgs::msg::Header{},
    std::vector<ManagedTensor>{
      ManagedTensor("images", TensorDataType::kFloat32, {1, 3, 640, 640}, buffer)});
  Transactions observations;
  ObservedAllocator allocator(*io->device_output_allocator(), observations);
  std::vector<inference::OutputTensor> outputs;
  {
    inference::OnnxInferenceCore core(cfg);
    outputs = core.RunInference(
      ManagedTensorBundleView(input), inference::OutputPlacement::kDevice, nullptr, &allocator);
  }
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(outputs[0].name, "output0");
  EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 84, 8400}));
  auto output = std::get<std::shared_ptr<DeviceBuffer>>(outputs[0].storage);
  EXPECT_EQ(output->get_blocking_ready_lease().data(), observations.pointers[0]);
  EXPECT_TRUE(output->attachment());
  float value;
  EXPECT_EQ(cudaMemcpy(&value, observations.pointers[0], sizeof(value), cudaMemcpyDeviceToHost),
    cudaSuccess);
}
INSTANTIATE_TEST_SUITE_P(
  NativeAndManaged, NativeOutputTest, ::testing::Values("tensor_list", "managed"));
} // namespace
