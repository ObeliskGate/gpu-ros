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

#include <gtest/gtest.h>
#ifdef GPU_ROS_TEST_CUDA
#include <cuda_runtime_api.h>
#include "cuda_buffer/cuda_buffer_api.hpp"
#endif
#ifdef GPU_ROS_TEST_TENSOR_LIST
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#endif

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "onnx_binding_test_peer.hpp"
#include <cstring>
namespace inference = gpu_ros::onnx_inference;
using Peer = inference::OnnxBindingTestPeer;
using Message = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

namespace
{
#ifdef GPU_ROS_TEST_CUDA

void Cuda(cudaError_t error)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

// A real CUDA stream is blocked by a host callback, with actual device work and
// an event queued behind it. No sleep or timing assumption establishes liveness.
class PendingGpuWork
{
public:
  PendingGpuWork()
  {
    Cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    try {
      Cuda(cudaEventCreateWithFlags(&finished_, cudaEventDisableTiming));
    } catch (...) {
      cudaStreamDestroy(stream_);
      throw;
    }
  }
  ~PendingGpuWork()
  {
    Release();
    cudaStreamSynchronize(stream_);
    cudaEventDestroy(finished_);
    cudaStreamDestroy(stream_);
  }
  void Submit(void * pointer)
  {
    try {
      Cuda(cudaLaunchHostFunc(stream_, [](void * context) {
        auto & self = *static_cast<PendingGpuWork *>(context);
        std::unique_lock<std::mutex> lock(self.mutex_);
        self.entered_ = true;
        self.cv_.notify_all();
        self.cv_.wait(lock, [&self] { return self.release_; });
      }, this));
      Cuda(cudaMemsetAsync(pointer, 0, 4 * sizeof(float), stream_));
      Cuda(cudaEventRecord(finished_, stream_));
    } catch (...) {
      Release();
      throw;
    }
  }
  void WaitUntilEntered()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return entered_; });
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    release_ = true;
    cv_.notify_all();
  }
  cudaError_t Query() { return cudaEventQuery(finished_); }
  void Finish() { Cuda(cudaEventSynchronize(finished_)); }

private:
  cudaStream_t stream_{};
  cudaEvent_t finished_{};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_{false};
  bool release_{false};
};
#endif

class BufferInferenceTest : public ::testing::TestWithParam<inference::ExecutionProvider>
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
    Peer::Reset();
#ifdef GPU_ROS_TEST_CUDA
    if (GetParam() == inference::ExecutionProvider::kCuda) {
      int count = 0;
      ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
      ASSERT_GT(count, 0);
    }
#endif
    allocator = inference::CreateBufferOutputAllocator(GetParam(), 0);
  }
  void TearDown() override { Peer::Reset(); allocator.reset(); rclcpp::shutdown(); }
  inference::OnnxInferenceCore::Config Config(bool direct = true)
  {
    inference::OnnxInferenceCore::Config cfg;
    cfg.model_file_path = TEST_DYNAMIC_OUTPUT_MODEL_PATH;
    cfg.ep = GetParam();
    cfg.transport = "rosidl_buffer";
    if (direct) { cfg.output_contracts = {"result=float32[1,4]"}; }
    return cfg;
  }
  std::shared_ptr<Message> Input(std::string name = "values", std::vector<int64_t> shape = {1, 4},
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  {
    auto message = std::make_shared<Message>();
    message->header.frame_id = "per_frame";
    message->header.stamp.sec = 17;
    message->header.stamp.nanosec = 42;
    auto & tensor = message->tensors.emplace_back();
    tensor.name = std::move(name);
    tensor.shape = std::move(shape);
    tensor.data_type = inference::OnnxToBundleDtype(dtype);
    const size_t bytes = inference::TensorByteSize(dtype, tensor.shape, tensor.name);
    std::vector<uint8_t> data(bytes);
    const float values[]{2, 4, 6, 8};
    std::memcpy(data.data(), values, std::min(bytes, sizeof(values)));
#ifdef GPU_ROS_TEST_CUDA
    if (GetParam() == inference::ExecutionProvider::kCuda) {
      tensor.data = cuda_buffer_backend::allocate_buffer(bytes);
      auto writer = cuda_buffer_backend::from_output_buffer(tensor.data, nullptr);
      Cuda(cudaMemcpy(writer.get_ptr(), data.data(), bytes, cudaMemcpyHostToDevice));
    } else
#endif
    { tensor.data = std::move(data); }
    return message;
  }
  std::vector<inference::OutputTensor> Run(inference::OnnxInferenceCore & core,
    std::shared_ptr<Message> input = {})
  {
    if (!input) { input = Input(); }
    return core.RunInference(inference::BindTensorBundle(input, GetParam(), 0),
      GetParam() == inference::ExecutionProvider::kCpu ? inference::OutputPlacement::kHost :
      inference::OutputPlacement::kDevice, nullptr, allocator.get());
  }
  void ExpectPayload(const std::vector<inference::OutputTensor> & output)
  {
    ASSERT_EQ(output.size(), 1U);
    EXPECT_EQ(output[0].name, "result");
    EXPECT_EQ(output[0].shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(output[0].dtype, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    auto message = inference::BufferOutputMessage(output);
    EXPECT_EQ(message->header.frame_id, "per_frame");
    EXPECT_EQ(message->header.stamp.sec, 17);
    EXPECT_EQ(message->header.stamp.nanosec, 42U);
    auto bytes = message->tensors[0].data.to_vector();
    ASSERT_EQ(bytes.size(), 4 * sizeof(float));
    float values[4];
    std::memcpy(values, bytes.data(), bytes.size());
    EXPECT_EQ((std::vector<float>(values, values + 4)), (std::vector<float>{3, 5, 7, 9}));
  }
  std::unique_ptr<inference::DeviceOutputAllocator> allocator;
};
TEST_P(BufferInferenceTest, DirectPointersAndPayloadOutliveSessionAndAllocator)
{
  std::vector<inference::OutputTensor> output;
  {
    inference::OnnxInferenceCore core(Config());
    output = Run(core);
    ASSERT_EQ(Peer::input_records.size(), 1U);
    EXPECT_TRUE(Peer::input_records[0].pointer_identity);
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
    EXPECT_EQ(std::get<inference::TensorStorage>(output[0].storage).data, Peer::output_pointer);
    EXPECT_TRUE(Peer::input_owner.expired());
  }
  EXPECT_TRUE(Peer::session_owner.expired());
  allocator.reset();
  ExpectPayload(output);
}
TEST_P(BufferInferenceTest, DynamicOutputCopiesActualShapeAndPayload)
{
  std::vector<inference::OutputTensor> output;
  {
    inference::OnnxInferenceCore core(Config(false));
    output = Run(core);
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_FALSE(Peer::output_records[0].pointer_identity);
    EXPECT_NE(Peer::output_records[0].pointer, Peer::output_records[0].ort_pointer);
  }
  allocator.reset();
  ExpectPayload(output);
}
TEST_P(BufferInferenceTest, ContractsRejectInvalidBindingsWithoutFallback)
{
  for (const auto & contract : {"result=int64[1,4]", "missing=float32[1,4]",
      "result=float32[0,4]", "result=float32[1,9223372036854775807]"}) {
    auto cfg = Config();
    cfg.output_contracts = {contract};
    EXPECT_ANY_THROW(inference::OnnxInferenceCore core(cfg));
  }
  auto cfg = Config();
  cfg.output_contracts = {"result=float32[1,4]", "result=float32[1,4]"};
  EXPECT_THROW(inference::OnnxInferenceCore core(cfg), std::invalid_argument);
  cfg.output_contracts = {"result=float32[1,5]"};
  inference::OnnxInferenceCore core(cfg);
  EXPECT_ANY_THROW(Run(core));
  EXPECT_TRUE(core.healthy());
}
TEST_P(BufferInferenceTest, InputNameRankDtypeAndCountRejectBeforeSubmission)
{
  inference::OnnxInferenceCore core(Config());
  EXPECT_THROW(Run(core, Input("wrong")), std::invalid_argument);
  EXPECT_THROW(Run(core, Input("values", {4})), std::invalid_argument);
  EXPECT_THROW(Run(core, Input("values", {2, 2})), std::invalid_argument);
  EXPECT_THROW(Run(core, Input("values", {1, 4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)), std::invalid_argument);
  auto extra = Input();
  extra->tensors.emplace_back();
  EXPECT_ANY_THROW(Run(core, extra));
  EXPECT_EQ(core.successful_runs(), 0U);
  EXPECT_TRUE(core.healthy());
  ExpectPayload(Run(core));
}
TEST_P(BufferInferenceTest, PreSubmissionCancellationReclaimsOwners)
{
  inference::OnnxInferenceCore core(Config());
  Peer::point = Peer::Point::kBeforeRun;
  EXPECT_THROW(Run(core), std::runtime_error);
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_TRUE(Peer::output_owner.expired());
  EXPECT_TRUE(core.healthy());
  ExpectPayload(Run(core));
}
TEST_P(BufferInferenceTest, SynchronizedFailureDoesNotQuarantine)
{
  inference::OnnxInferenceCore core(Config());
  Peer::point = Peer::Point::kSynchronized;
  EXPECT_THROW(Run(core), std::runtime_error);
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_TRUE(Peer::output_owner.expired());
  EXPECT_TRUE(core.healthy());
  ExpectPayload(Run(core));
}
TEST_P(BufferInferenceTest, ReportFailureDoesNotQuarantine)
{
  auto cfg = Config();
  cfg.binding_report_path = "/dev/null/binding.json";
  inference::OnnxInferenceCore core(cfg);
  EXPECT_ANY_THROW(Run(core));
  EXPECT_TRUE(core.healthy());
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_TRUE(Peer::output_owner.expired());
  EXPECT_EQ(core.successful_runs(), 1U);
}
TEST(BufferAdapterTest, ValidatesOffsetCapacityStridesAndOwner)
{
  auto buffer = std::make_shared<rosidl::Buffer<uint8_t>>(std::vector<uint8_t>(20));
  auto access = gpu_ros::rosidl_buffer::CreateCpuBufferAccess();
  auto memory = inference::ProviderMemoryInfo(inference::ExecutionProvider::kCpu, 0);
  auto bind = [&](size_t offset, std::vector<int64_t> strides) {
    return inference::BindBuffer(*buffer, "values", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      {1, 4}, offset, strides, buffer, *access, memory);
  };
  auto value = bind(4, {4, 1});
  EXPECT_EQ(value.data, buffer->data() + 4);
  EXPECT_THROW(bind(8, {}), std::invalid_argument);
  EXPECT_THROW(bind(1, {}), std::invalid_argument);
  EXPECT_THROW(bind(0, {4, 2}), std::invalid_argument);
  EXPECT_THROW(bind(0, {1}), std::invalid_argument);
  std::weak_ptr<const void> owner = buffer;
  buffer.reset();
  EXPECT_FALSE(owner.expired());
  value.value = Ort::Value{nullptr};
  value.owner.reset();
  EXPECT_TRUE(owner.expired());
}
#ifdef GPU_ROS_TEST_TENSOR_LIST
TEST(TensorListBufferIOTest, MixedCpuAndCudaBuffersRetainTheirOriginalStorage)
{
  if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  auto node = std::make_shared<rclcpp::Node>("mixed_buffer_wire_test", options);
  node->declare_parameter<std::string>("execution_provider", "cuda");
  node->declare_parameter<int>("gpu_device_id", 0);
  node->declare_parameter<std::string>("message_format", "tensor_list");
  auto io = inference::CreateTensorBundleIO(node.get(), "rosidl_buffer", false);
  std::promise<inference::TensorBindingBatch> received;
  auto future = received.get_future();
  io->Subscribe([&received](inference::TensorBindingBatch input) {
    received.set_value(std::move(input));
  });
  using WireMessage = isaac_ros_tensor_msgs::msg::TensorList;
  auto publisher = node->create_publisher<WireMessage>("tensor_input", rclcpp::QoS(10));
  auto message = std::make_unique<WireMessage>();
  const auto * original_message = message.get();
  message->names = {"images", "orig_target_sizes"};
  message->tensors.resize(2);
  auto & image = message->tensors[0];
  image.dtype_code = 2;
  image.dtype_bits = 32;
  image.dtype_lanes = 1;
  image.shape = {1, 4};
  image.strides = {4, 1};
  image.byte_offset = 0;
  const float pixels[]{2, 4, 6, 8};
  image.data = cuda_buffer_backend::allocate_buffer(sizeof(pixels));
  const void * original_device_pointer;
  {
    auto writer = cuda_buffer_backend::from_output_buffer(image.data, nullptr);
    original_device_pointer = writer.get_ptr();
    ASSERT_EQ(cudaMemcpy(writer.get_ptr(), pixels, sizeof(pixels), cudaMemcpyHostToDevice), cudaSuccess);
  }
  auto & sizes = message->tensors[1];
  sizes.dtype_code = 0;
  sizes.dtype_bits = 64;
  sizes.dtype_lanes = 1;
  sizes.shape = {1, 2};
  sizes.strides = {2, 1};
  sizes.byte_offset = 0;
  const int64_t dimensions[]{720, 1280};
  std::vector<uint8_t> bytes(sizeof(dimensions));
  std::memcpy(bytes.data(), dimensions, sizeof(dimensions));
  sizes.data = std::move(bytes);
  const void * original_host_pointer = sizes.data.data();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  publisher->publish(std::move(message));
  ASSERT_EQ(executor.spin_until_future_complete(future, std::chrono::seconds(5)),
    rclcpp::FutureReturnCode::SUCCESS);
  auto input = future.get();
  ASSERT_EQ(input.bindings.size(), 2U);
  EXPECT_EQ(input.owner.get(), original_message);
  EXPECT_EQ(input.bindings[0].data, original_device_pointer);
  EXPECT_EQ(input.bindings[0].value.GetTensorMemoryInfo().GetDeviceType(),
    OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(input.bindings[1].data, original_host_pointer);
  EXPECT_EQ(input.bindings[1].value.GetTensorMemoryInfo().GetDeviceType(),
    OrtMemoryInfoDeviceType_CPU);
  executor.remove_node(node);
  io.reset();
  publisher.reset();
  node.reset();
  float actual_pixels[4];
  ASSERT_EQ(cudaMemcpy(actual_pixels, input.bindings[0].data, sizeof(actual_pixels),
    cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ((std::vector<float>(actual_pixels, actual_pixels + 4)),
    (std::vector<float>{2, 4, 6, 8}));
  int64_t actual_dimensions[2];
  std::memcpy(actual_dimensions, input.bindings[1].data, sizeof(actual_dimensions));
  EXPECT_EQ((std::vector<int64_t>(actual_dimensions, actual_dimensions + 2)),
    (std::vector<int64_t>{720, 1280}));
  rclcpp::shutdown();
}
#endif
TEST(BufferNodeTest, RejectsInvalidParametersBeforeSubscribing)
{
  if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
  for (const auto & parameter : {
      rclcpp::Parameter("gpu_device_id", -1),
      rclcpp::Parameter("ort_profile_frames", -1),
      rclcpp::Parameter("ort_profile_frames", 1),
      rclcpp::Parameter("output_pool_capacity", 0)}) {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("execution_provider", "cpu"),
      rclcpp::Parameter("transport", "rosidl_buffer"), parameter});
    EXPECT_THROW(inference::OnnxInferenceNode node(options), std::invalid_argument);
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("execution_provider", "cpu"),
    rclcpp::Parameter("transport", "rosidl_buffer"),
    rclcpp::Parameter("model_file_path", "/missing/buffer-model.onnx")});
  EXPECT_ANY_THROW(inference::OnnxInferenceNode node(options));
  rclcpp::shutdown();
}
TEST(BufferNodeTest, ShutdownDrainsActiveCallbackAndRejectsQueuedCallbacks)
{
  if (!rclcpp::ok()) { rclcpp::init(0, nullptr); }
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("execution_provider", "cpu"),
    rclcpp::Parameter("transport", "rosidl_buffer")});
  auto node = std::make_unique<inference::OnnxInferenceNode>(options);
  auto callback = Peer::Callback(*node);
  auto state = Peer::State(*node);
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  std::atomic<size_t> calls{0};
  Peer::callback_entered = [&] { ++calls; entered.set_value(); released.wait(); };
  auto active = std::async(std::launch::async, [&] { callback({}); });
  entered.get_future().wait();
  auto shutdown = std::async(std::launch::async, [&] { node.reset(); });
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [&] { return state->shutting_down; });
    EXPECT_EQ(state->active_callbacks, 1U);
  }
  callback({});
  EXPECT_EQ(calls.load(), 1U);
  release.set_value();
  active.get();
  shutdown.get();
  callback({});
  EXPECT_EQ(calls.load(), 1U);
  Peer::callback_entered = {};
  rclcpp::shutdown();
}
#ifdef GPU_ROS_TEST_CUDA
TEST_P(BufferInferenceTest, SubmittedFailureRetainsOwnersThroughRealGpuCompletion)
{
  if (GetParam() != inference::ExecutionProvider::kCuda) { return; }
  inference::OnnxInferenceCore core(Config());
  PendingGpuWork work;
  std::promise<void> submitted;
  std::weak_ptr<const void> input_owner, output_owner;
  auto run = std::async(std::launch::async, [&] {
    bool signalled = false;
    Peer::Reset();
    Peer::point = Peer::Point::kSubmissionBoundary;
    Peer::submission_work = [&](void * pointer) {
      input_owner = Peer::input_owner;
      output_owner = Peer::output_owner;
      work.Submit(pointer);
      submitted.set_value();
      signalled = true;
    };
    try {
      Run(core);
      if (!signalled) { submitted.set_exception(std::make_exception_ptr(std::runtime_error("submission hook not reached"))); }
      return false;
    } catch (...) {
      if (!signalled) { submitted.set_exception(std::current_exception()); }
      return true;
    }
  });
  submitted.get_future().get();
  work.WaitUntilEntered();
  EXPECT_EQ(work.Query(), cudaErrorNotReady);
  EXPECT_FALSE(input_owner.expired());
  EXPECT_FALSE(output_owner.expired());
  work.Release();
  EXPECT_TRUE(run.get());
  work.Finish();
  EXPECT_TRUE(input_owner.expired());
  EXPECT_TRUE(output_owner.expired());
  EXPECT_TRUE(core.healthy());
  ExpectPayload(Run(core));
}
TEST_P(BufferInferenceTest, UnknownCompletionRetainsFullRequestAndPoisonsCore)
{
  if (GetParam() != inference::ExecutionProvider::kCuda) { return; }
  auto queued = inference::BindTensorBundle(Input(), GetParam(), 0);
  PendingGpuWork work;
  std::weak_ptr<inference::OnnxSession> session;
  std::weak_ptr<const void> input, output;
  {
    inference::OnnxInferenceCore core(Config());
    Peer::point = Peer::Point::kSubmissionBoundary;
    Peer::unknown_completion = true;
    Peer::submission_work = [&](void * pointer) { work.Submit(pointer); };
    EXPECT_THROW(Run(core), std::runtime_error);
    work.WaitUntilEntered();
    EXPECT_EQ(work.Query(), cudaErrorNotReady);
    input = Peer::input_owner;
    output = Peer::output_owner;
    session = Peer::session_owner;
    EXPECT_FALSE(core.healthy());
    EXPECT_THROW(core.RunInference(std::move(queued), inference::OutputPlacement::kDevice,
      nullptr, allocator.get()), std::runtime_error);
  }
  work.Release();
  work.Finish();
  EXPECT_FALSE(input.expired());
  EXPECT_FALSE(output.expired());
  EXPECT_FALSE(session.expired());
}
TEST_P(BufferInferenceTest, RealYoloUsesFormalDirectOutputContract)
{
  if (GetParam() != inference::ExecutionProvider::kCuda) { return; }
  auto cfg = Config();
  cfg.model_file_path = TEST_CUDA_IO_BINDING_MODEL_PATH;
  cfg.output_contracts = {"output0=float32[1,84,8400]"};
  std::vector<inference::OutputTensor> output;
  {
    inference::OnnxInferenceCore core(cfg);
    output = Run(core, Input("images", {1, 3, 640, 640}));
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
  }
  allocator.reset();
  auto message = inference::BufferOutputMessage(output);
  EXPECT_EQ(message->tensors[0].shape, (std::vector<int64_t>{1, 84, 8400}));
  auto bytes = message->tensors[0].data.to_vector();
  ASSERT_EQ(bytes.size(), 84U * 8400U * sizeof(float));
  for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
    float value;
    std::memcpy(&value, bytes.data() + i, sizeof(float));
    ASSERT_TRUE(std::isfinite(value));
  }
}
INSTANTIATE_TEST_SUITE_P(CpuAndCuda, BufferInferenceTest,
  ::testing::Values(inference::ExecutionProvider::kCpu, inference::ExecutionProvider::kCuda));
#else
INSTANTIATE_TEST_SUITE_P(Cpu, BufferInferenceTest, ::testing::Values(inference::ExecutionProvider::kCpu));
#endif

} // namespace
