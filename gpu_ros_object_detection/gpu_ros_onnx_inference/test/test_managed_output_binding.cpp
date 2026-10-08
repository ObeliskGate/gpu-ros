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
#include <cmath>
#include <cstring>
#include "gpu_ros_onnx_inference/managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "onnx_binding_test_peer.hpp"
#ifdef GPU_ROS_TEST_CUDA
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif
#ifdef GPU_ROS_TEST_HIP
#include "gpu_ros_managed_hip/hip_backend.hpp"
#endif
namespace inference = gpu_ros::onnx_inference;
using namespace gpu_ros_managed;
using Peer = inference::OnnxBindingTestPeer;
namespace
{
struct Observations
{
  size_t cancelled{0}, failed{0}, completed{0};
  bool fail_binding{false};
  size_t fail_index{0};
  std::vector<void *> pointers;
};
class ObservedBatch final : public inference::DeviceOutputBatch
{
public:
  ObservedBatch(std::unique_ptr<inference::DeviceOutputBatch> batch, Observations & observations)
  : batch_(std::move(batch)), observations_(observations) {}
  size_t size() const noexcept override { return batch_->size(); }
  void * pointer(size_t i) const override
  {
    if (observations_.fail_binding && i == observations_.fail_index) {
      throw std::runtime_error("binding failure");
    }
    return batch_->pointer(i);
  }
  inference::TensorStorage storage(size_t i) const override { return batch_->storage(i); }
  void RetainOwner(std::shared_ptr<const void> owner) override { batch_->RetainOwner(std::move(owner)); }
  void CompleteAfterSync() override { batch_->CompleteAfterSync(); ++observations_.completed; }
  void CancelBeforeSubmit() noexcept override { batch_->CancelBeforeSubmit(); ++observations_.cancelled; }
  void FailAfterSubmit() noexcept override { batch_->FailAfterSubmit(); ++observations_.failed; }
  void CopyFrom(const std::vector<inference::OutputTensor> & output) override { batch_->CopyFrom(output); }
private:
  std::unique_ptr<inference::DeviceOutputBatch> batch_;
  Observations & observations_;
};
class ObservedAllocator final : public inference::DeviceOutputAllocator
{
public:
  ObservedAllocator(inference::DeviceOutputAllocator & allocator, Observations & observations)
  : allocator_(allocator), observations_(observations) {}
  std::unique_ptr<inference::DeviceOutputBatch> Allocate(const std_msgs::msg::Header & header,
    const std::vector<inference::DeviceOutputSpec> & specs) override
  {
    auto batch = allocator_.Allocate(header, specs);
    for (size_t i = 0; i < batch->size(); ++i) { observations_.pointers.push_back(batch->pointer(i)); }
    return std::make_unique<ObservedBatch>(std::move(batch), observations_);
  }
  Ort::MemoryInfo memory_info() const override { return allocator_.memory_info(); }
  bool adopts_dynamic_outputs() const noexcept override { return allocator_.adopts_dynamic_outputs(); }
  bool permits_preallocation_fallback() const noexcept override
  { return allocator_.permits_preallocation_fallback(); }
  inference::TensorStorage Adopt(inference::TensorStorage storage) override
  { return allocator_.Adopt(std::move(storage)); }
  bool SynchronizeAfterFailure() noexcept override { return allocator_.SynchronizeAfterFailure(); }
private:
  inference::DeviceOutputAllocator & allocator_;
  Observations & observations_;
};
class ManagedOutputTest : public ::testing::TestWithParam<inference::ExecutionProvider>
{
protected:
  void SetUp() override
  {
    Peer::Reset();
#ifdef GPU_ROS_TEST_CUDA
    if (GetParam() == inference::ExecutionProvider::kCuda) {
      int count = 0;
      ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
      ASSERT_GT(count, 0);
    }
#endif
#ifdef GPU_ROS_TEST_HIP
    if (GetParam() == inference::ExecutionProvider::kMigraphx) {
      int count = 0;
      ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
      ASSERT_GT(count, 0);
    }
#endif
  }
  void TearDown() override { Peer::Reset(); }
  inference::OnnxInferenceCore::Config Config(bool direct = true, bool strict = false)
  {
    inference::OnnxInferenceCore::Config cfg;
    cfg.model_file_path = TEST_DYNAMIC_OUTPUT_MODEL_PATH;
    cfg.ep = GetParam();
    cfg.transport = "managed";
    cfg.output_pool_capacity = 1;
    cfg.output_pool_wait_timeout = std::chrono::milliseconds(0);
    if (direct) { cfg.output_contracts = {"result=float32[1,4]"}; }
    if (strict) {
      cfg.io_contract = "device_strict";
      cfg.input_contracts = {"values=float32[1,4]"};
    }
    return cfg;
  }
  std::shared_ptr<DeviceBuffer> Allocate(size_t bytes)
  {
#ifdef GPU_ROS_TEST_CUDA
    if (GetParam() == inference::ExecutionProvider::kCuda) { return cuda::allocate(bytes, 0); }
#endif
#ifdef GPU_ROS_TEST_HIP
    if (GetParam() == inference::ExecutionProvider::kMigraphx) { return hip::allocate(bytes, 0); }
#endif
    throw std::runtime_error("test backend not built");
  }
  inference::TensorBindingBatch Input()
  {
    const float values[]{2, 4, 6, 8};
    auto buffer = Allocate(sizeof(values));
    buffer->copy_from_host_blocking(values, sizeof(values));
    auto message = std::make_shared<ManagedTensorBundle>(std_msgs::msg::Header{},
      std::vector<ManagedTensor>{ManagedTensor("values", TensorDataType::kFloat32, {1, 4}, buffer)});
    return inference::BindManagedTensorBundle(ManagedTensorBundleView(message), GetParam(), 0);
  }
  void ExpectPayload(const std::vector<inference::OutputTensor> & outputs)
  {
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].shape, (std::vector<int64_t>{1, 4}));
    auto storage = std::get<inference::TensorStorage>(outputs[0].storage);
    auto buffer = std::static_pointer_cast<DeviceBuffer>(storage.envelope);
    EXPECT_EQ(buffer->get_blocking_ready_lease().data(), storage.data);
    float values[4]{};
    buffer->copy_to_host_blocking(values, sizeof(values));
    EXPECT_EQ((std::vector<float>(values, values + 4)), (std::vector<float>{3, 5, 7, 9}));
  }
};
TEST_P(ManagedOutputTest, DirectPointersAndPayloadOutliveSessionAndAllocator)
{
  std::vector<inference::OutputTensor> output;
  Observations observed;
  {
    auto cfg = Config();
    auto base = inference::CreateManagedOutputAllocator(cfg);
    ObservedAllocator allocator(*base, observed);
    inference::OnnxInferenceCore core(cfg);
    output = core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, &allocator);
    ASSERT_EQ(output.size(), 1U);
    EXPECT_EQ(std::get<inference::TensorStorage>(output[0].storage).data, observed.pointers[0]);
    EXPECT_EQ(observed.completed, 1U);
  }
  ExpectPayload(output);
}
TEST_P(ManagedOutputTest, DynamicOutputAdoptsOrtAllocationAndKeepsSessionAlive)
{
  std::vector<inference::OutputTensor> output;
  std::weak_ptr<inference::OnnxSession> session;
  {
    auto cfg = Config(false);
    auto allocator = inference::CreateManagedOutputAllocator(cfg);
    inference::OnnxInferenceCore core(cfg);
    output = core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, allocator.get());
    session = Peer::session_owner;
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
  }
  EXPECT_FALSE(session.expired());
  ExpectPayload(output);
  output.clear();
  EXPECT_TRUE(session.expired());
}
TEST_P(ManagedOutputTest, PreSubmissionCancellationRestoresStrictPool)
{
  auto cfg = Config(true, true);
  auto base = inference::CreateManagedOutputAllocator(cfg);
  Observations observed;
  observed.fail_binding = true;
  ObservedAllocator allocator(*base, observed);
  inference::OnnxInferenceCore core(cfg);
  EXPECT_THROW(core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, &allocator), std::runtime_error);
  EXPECT_EQ(observed.cancelled, 1U);
  EXPECT_EQ(observed.failed, 0U);
  EXPECT_TRUE(core.healthy());
  observed.fail_binding = false;
  auto outputs = core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, &allocator);
  ExpectPayload(outputs);
  outputs.clear();
  EXPECT_TRUE(base->shutdown(std::chrono::seconds(5)));
}
TEST_P(ManagedOutputTest, ExhaustedStrictPoolNeverFallsBack)
{
  auto cfg = Config(true, true);
  auto allocator = inference::CreateManagedOutputAllocator(cfg);
  inference::OnnxInferenceCore core(cfg);
  auto held = core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, allocator.get());
  EXPECT_THROW(core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, allocator.get()), std::runtime_error);
  EXPECT_EQ(core.successful_runs(), 1U);
  held.clear();
  ExpectPayload(core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, allocator.get()));
}
TEST_P(ManagedOutputTest, UnknownSubmissionFailsReservationsAndPoisonsCore)
{
  auto cfg = Config(true, true);
  auto base = inference::CreateManagedOutputAllocator(cfg);
  Observations observed;
  ObservedAllocator allocator(*base, observed);
  inference::OnnxInferenceCore core(cfg);
  Peer::point = Peer::Point::kSubmissionBoundary;
  Peer::unknown_completion = true;
  EXPECT_THROW(core.RunInference(Input(), inference::OutputPlacement::kDevice, nullptr, &allocator), std::runtime_error);
  EXPECT_EQ(observed.failed, 1U);
  EXPECT_EQ(observed.cancelled, 0U);
  EXPECT_FALSE(core.healthy());
  EXPECT_FALSE(Peer::input_owner.expired());
  EXPECT_FALSE(Peer::output_owner.expired());
  EXPECT_FALSE(base->shutdown(std::chrono::milliseconds(0)));
}
TEST_P(ManagedOutputTest, ContractMismatchIsRejected)
{
  auto cfg = Config();
  cfg.output_contracts = {"result=int64[1,4]"};
  EXPECT_THROW(inference::OnnxInferenceCore core(cfg), std::invalid_argument);
  cfg.output_contracts = {"missing=float32[1,4]"};
  EXPECT_THROW(inference::OnnxInferenceCore core(cfg), std::invalid_argument);
}
#ifdef BUILD_NATIVE_TENSOR_LIST_TRANSPORT
TEST_P(ManagedOutputTest, NvidiaManagedIoPreservesFinalWireAllocationAttachment)
{
  if (GetParam() != inference::ExecutionProvider::kCuda) { return; }
  if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
  std::shared_ptr<ManagedTensorBundle> published;
  std::shared_ptr<DeviceBuffer> original_buffer;
  Observations observed;
  std_msgs::msg::Header header;
  header.stamp.sec = 17;
  header.stamp.nanosec = 42;
  header.frame_id = "managed_wire_owner";
  {
    auto node = std::make_shared<rclcpp::Node>("managed_wire_output_test");
    node->declare_parameter<std::string>("execution_provider", "cuda");
    node->declare_parameter<int>("gpu_device_id", 0);
    auto io = inference::CreateTensorBundleIO(node.get(), "managed", false);
    ASSERT_NE(io->device_output_allocator(), nullptr);
    ObservedAllocator allocator(*io->device_output_allocator(), observed);
    inference::OnnxInferenceCore core(Config());
    auto input = Input();
    input.header = header;
    auto outputs = core.RunInference(std::move(input), inference::OutputPlacement::kDevice,
      nullptr, &allocator);
    ASSERT_EQ(outputs.size(), 1U);
    const auto & storage = std::get<inference::TensorStorage>(outputs[0].storage);
    original_buffer = std::static_pointer_cast<DeviceBuffer>(storage.envelope);
    ASSERT_TRUE(original_buffer);
    ASSERT_TRUE(original_buffer->attachment());
    EXPECT_EQ(storage.data, observed.pointers.at(0));
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
    published = std::make_shared<ManagedTensorBundle>(
      inference::ManagedOutputMessage({header, std::move(outputs)}));
    ASSERT_EQ(published->tensors().size(), 1U);
    EXPECT_EQ(std::get<std::shared_ptr<DeviceBuffer>>(published->tensors()[0].storage()),
      original_buffer);
    EXPECT_EQ(observed.completed, 1U);
  }
  // The exact attachment-bearing DeviceBuffer survives conversion and the
  // allocator/node/session. Compat's existing reuse path receives this object,
  // not a fresh plain allocation requiring an output-sized D2D copy.
  EXPECT_EQ(published->header(), header);
  EXPECT_EQ(published->tensors()[0].name(), "result");
  EXPECT_EQ(published->tensors()[0].shape(), (std::vector<int64_t>{1, 4}));
  EXPECT_EQ(published->tensors()[0].data_type(), TensorDataType::kFloat32);
  EXPECT_EQ(original_buffer->get_blocking_ready_lease().data(), observed.pointers.at(0));
  ASSERT_TRUE(original_buffer->attachment());
  float actual[4];
  original_buffer->copy_to_host_blocking(actual, sizeof(actual));
  EXPECT_EQ((std::vector<float>(actual, actual + 4)), (std::vector<float>{3, 5, 7, 9}));
  published.reset();
  original_buffer.reset();
  rclcpp::shutdown();
}
#endif
#ifdef GPU_ROS_TEST_CUDA
TEST_P(ManagedOutputTest, RealYoloDirectOutputRetainsFullPayload)
{
  if (GetParam() != inference::ExecutionProvider::kCuda) { return; }
  auto cfg = Config();
  cfg.model_file_path = TEST_CUDA_IO_BINDING_MODEL_PATH;
  cfg.output_contracts = {"output0=float32[1,84,8400]"};
  auto input = Allocate(3U * 640U * 640U * sizeof(float));
  std::vector<float> zeros(3U * 640U * 640U, 0);
  input->copy_from_host_blocking(zeros.data(), zeros.size() * sizeof(float));
  auto message = std::make_shared<ManagedTensorBundle>(std_msgs::msg::Header{},
    std::vector<ManagedTensor>{ManagedTensor("images", TensorDataType::kFloat32, {1, 3, 640, 640}, input)});
  std::vector<inference::OutputTensor> outputs;
  {
    auto allocator = inference::CreateManagedOutputAllocator(cfg);
    inference::OnnxInferenceCore core(cfg);
    outputs = core.RunInference(inference::BindManagedTensorBundle(ManagedTensorBundleView(message), GetParam(), 0),
      inference::OutputPlacement::kDevice, nullptr, allocator.get());
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
  }
  auto buffer = std::static_pointer_cast<DeviceBuffer>(std::get<inference::TensorStorage>(outputs[0].storage).envelope);
  std::vector<float> values(84U * 8400U);
  buffer->copy_to_host_blocking(values.data(), values.size() * sizeof(float));
  for (float value : values) { ASSERT_TRUE(std::isfinite(value)); }
}
#endif
#ifdef GPU_ROS_TEST_HIP
TEST_P(ManagedOutputTest, HipThreeOutputCancellationAndNumericPayload)
{
  if (GetParam() != inference::ExecutionProvider::kMigraphx) { return; }
  const std::vector<float> pixels(12, 0.25F);
  const std::vector<int64_t> sizes{2, 2};
  auto images = Allocate(pixels.size() * sizeof(float));
  images->copy_from_host_blocking(pixels.data(), pixels.size() * sizeof(float));
  auto target_sizes = Allocate(sizes.size() * sizeof(int64_t));
  target_sizes->copy_from_host_blocking(sizes.data(), sizes.size() * sizeof(int64_t));
  auto message = std::make_shared<ManagedTensorBundle>(std_msgs::msg::Header{},
    std::vector<ManagedTensor>{
      ManagedTensor("images", TensorDataType::kFloat32, {1, 3, 2, 2}, images),
      ManagedTensor("orig_target_sizes", TensorDataType::kInt64, {1, 2}, target_sizes)});
  for (const auto scenario : {std::pair{false, 0U}, std::pair{false, 1U}, std::pair{false, 2U},
      std::pair{true, 0U}, std::pair{true, 1U}, std::pair{true, 2U}}) {
    const bool strict = scenario.first;
    auto cfg = Config(true, strict);
    cfg.model_file_path = TEST_HIP_CANCELLATION_MODEL_PATH;
    cfg.input_contracts = {"images=float32[1,3,2,2]", "orig_target_sizes=int64[1,2]"};
    cfg.output_contracts = {"labels=int64[1,300]", "boxes=float32[1,300,4]", "scores=float32[1,300]"};
    auto base = inference::CreateManagedOutputAllocator(cfg);
    Observations observed;
    ObservedAllocator allocator(*base, observed);
    inference::OnnxInferenceCore core(cfg);
    auto run = [&] {
      return core.RunInference(inference::BindManagedTensorBundle(ManagedTensorBundleView(message),
        GetParam(), 0, strict), inference::OutputPlacement::kDevice, nullptr, &allocator);
    };
    std::shared_ptr<DeviceBuffer> cancelled_buffer;
    observed.fail_binding = scenario.second != 2;
    observed.fail_index = scenario.second;
    if (scenario.second == 2) {
      Peer::reservation_hook = [&](size_t index, const inference::TensorStorage & storage) {
        if (index == 0) {
          cancelled_buffer = std::static_pointer_cast<DeviceBuffer>(storage.envelope);
          throw std::runtime_error("first writer acquisition failure");
        }
      };
    }
    if (strict) { EXPECT_THROW(run(), std::runtime_error); }
    else { EXPECT_EQ(run().size(), 3U); }
    EXPECT_EQ(observed.cancelled, scenario.second == 2 ? 0U : 1U);
    EXPECT_EQ(observed.failed, 0U);
    Peer::reservation_hook = {};
    if (cancelled_buffer) {
      EXPECT_FALSE(cancelled_buffer->failed());
      EXPECT_EQ(cancelled_buffer->readiness(), BufferReadiness::kNotReady);
      auto writer = cancelled_buffer->get_synchronized_write_handle();
      writer.cancel();
    }
    cancelled_buffer.reset();
    observed.fail_binding = false;
    auto outputs = run();
    ASSERT_EQ(outputs.size(), 3U);
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto buffer = std::static_pointer_cast<DeviceBuffer>(
        std::get<inference::TensorStorage>(outputs[i].storage).envelope);
      EXPECT_FALSE(buffer->failed());
      if (i == 0) {
        EXPECT_EQ(outputs[i].name, "labels");
        EXPECT_EQ(outputs[i].dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        EXPECT_EQ(outputs[i].shape, (std::vector<int64_t>{1, 300}));
        std::vector<int64_t> actual(300), expected(300, 0);
        expected[0] = 7;
        buffer->copy_to_host_blocking(actual.data(), actual.size() * sizeof(int64_t));
        EXPECT_EQ(actual, expected);
      } else {
        EXPECT_EQ(outputs[i].name, i == 1 ? "boxes" : "scores");
        EXPECT_EQ(outputs[i].dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        EXPECT_EQ(outputs[i].shape, i == 1 ? (std::vector<int64_t>{1, 300, 4}) :
          (std::vector<int64_t>{1, 300}));
        std::vector<float> actual(i == 1 ? 1200 : 300), expected(actual.size(), 0);
        if (i == 1) { expected[0] = 10; expected[1] = 20; expected[2] = 30; expected[3] = 50; }
        else { expected[0] = 0.95F; }
        buffer->copy_to_host_blocking(actual.data(), actual.size() * sizeof(float));
        for (size_t j = 0; j < actual.size(); ++j) { EXPECT_FLOAT_EQ(actual[j], expected[j]); }
      }
    }
    outputs.clear();
    EXPECT_TRUE(base->shutdown(std::chrono::seconds(5)));
  }
}
#endif
#if defined(GPU_ROS_TEST_CUDA) && defined(GPU_ROS_TEST_HIP)
INSTANTIATE_TEST_SUITE_P(DeviceBackends, ManagedOutputTest,
  ::testing::Values(inference::ExecutionProvider::kCuda, inference::ExecutionProvider::kMigraphx));
#elif defined(GPU_ROS_TEST_CUDA)
INSTANTIATE_TEST_SUITE_P(Cuda, ManagedOutputTest, ::testing::Values(inference::ExecutionProvider::kCuda));
#elif defined(GPU_ROS_TEST_HIP)
INSTANTIATE_TEST_SUITE_P(Hip, ManagedOutputTest, ::testing::Values(inference::ExecutionProvider::kMigraphx));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ManagedOutputTest);
#endif
TEST(ManagedHostAdapterTest, HostInputAliasesAndRetainsOriginalMessage)
{
  auto values = std::make_shared<std::vector<float>>(std::initializer_list<float>{2, 4, 6, 8});
  auto tensor = ManagedTensor::from_host_external("values", TensorDataType::kFloat32,
    {1, 4}, values, values->data(), 4 * sizeof(float));
  auto message = std::make_shared<ManagedTensorBundle>(std_msgs::msg::Header{},
    std::vector<ManagedTensor>{std::move(tensor)});
  auto input = inference::BindManagedTensorBundle(ManagedTensorBundleView(message),
    inference::ExecutionProvider::kCpu, 0);
  EXPECT_EQ(input.bindings[0].data, values->data());
  inference::OnnxInferenceCore::Config cfg;
  cfg.model_file_path = TEST_DYNAMIC_OUTPUT_MODEL_PATH;
  cfg.ep = inference::ExecutionProvider::kCpu;
  inference::OnnxInferenceCore core(cfg);
  auto output = core.RunInference(std::move(input));
  const auto & bytes = std::get<std::vector<uint8_t>>(output[0].storage);
  float actual[4];
  ASSERT_EQ(bytes.size(), sizeof(actual));
  std::memcpy(actual, bytes.data(), sizeof(actual));
  EXPECT_EQ((std::vector<float>(actual, actual + 4)), (std::vector<float>{3, 5, 7, 9}));
}
} // namespace
