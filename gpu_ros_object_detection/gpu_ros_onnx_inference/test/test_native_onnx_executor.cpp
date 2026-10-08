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
#include <cuda_runtime_api.h>

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

#include "native_onnx_executor.hpp"
#include "native_onnx_executor_test_peer.hpp"

namespace inference = gpu_ros::onnx_inference;
namespace native = gpu_ros::nvidia_tensor_bundle_compat::native;
using Peer = inference::NativeOnnxExecutorTestPeer;

namespace
{

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

class NativeExecutorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0) << "GPU-dependent native tests must not pass without a device";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    Peer::Reset();
    node_ = std::make_shared<rclcpp::Node>("native_executor_test");
    transport_ = std::make_unique<native::TensorListTransport>(node_.get(), 0, "tensor_output", false);
  }
  void TearDown() override
  {
    Peer::callback_entered = {};
    Peer::Reset();
    transport_.reset();
    node_.reset();
    rclcpp::shutdown();
  }
  inference::NativeOnnxExecutor::Config Config(bool direct = true)
  {
    inference::NativeOnnxExecutor::Config config;
    config.session = {TEST_DYNAMIC_OUTPUT_MODEL_PATH, inference::ExecutionProvider::kCuda, 0, ""};
    if (direct) {
      config.output_contracts = {"result=float32[1,4]"};
    }
    return config;
  }
  native::TensorList Input(const std::string & name = "values",
    const std::vector<int64_t> & shape = {1, 4})
  {
    auto values = std::make_shared<std::vector<float>>(std::initializer_list<float>{2, 4, 6, 8});
    std_msgs::msg::Header header;
    header.frame_id = "per_frame";
    header.stamp.sec = 17;
    header.stamp.nanosec = 42;
    auto batch = transport_->Allocate(header, {{name, 2, 32, 1, shape}});
    batch.CopyFrom({{values->data(), values->size() * sizeof(float), false}}, values);
    return batch.message();
  }
  void ExpectPayload(const native::TensorList & output)
  {
    ASSERT_EQ(output.tensors().size(), 1U);
    EXPECT_EQ(output.tensors()[0].name, "result");
    EXPECT_EQ(output.tensors()[0].shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(output.tensors()[0].dtype_code, 2U);
    EXPECT_EQ(output.tensors()[0].dtype_bits, 32U);
    EXPECT_EQ(output.tensors()[0].dtype_lanes, 1U);
    EXPECT_EQ(output.header().frame_id, "per_frame");
    EXPECT_EQ(output.header().stamp.sec, 17);
    EXPECT_EQ(output.header().stamp.nanosec, 42U);
    float values[4]{};
    ASSERT_EQ(cudaMemcpy(values, output.data(0), sizeof(values), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ((std::vector<float>(values, values + 4)), (std::vector<float>{3, 5, 7, 9}));
  }
  std::shared_ptr<rclcpp::Node> node_;
  std::unique_ptr<native::TensorListTransport> transport_;
};

TEST_F(NativeExecutorTest, DirectPointersAndPayloadOutliveSessionAndTransport)
{
  std::optional<native::TensorList> output;
  {
    inference::NativeOnnxExecutor executor(Config());
    std::optional<native::TensorList> input(Input());
    const auto input_pointer = input->data(0);
    output.emplace(executor.Run(*input, *transport_));
    input.reset();
    ASSERT_EQ(Peer::input_records.size(), 1U);
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_EQ(Peer::input_records[0].pointer, inference::PointerString(input_pointer));
    EXPECT_EQ(Peer::input_records[0].pointer, Peer::input_records[0].ort_pointer);
    EXPECT_TRUE(Peer::input_records[0].pointer_identity);
    EXPECT_EQ(output->data(0), Peer::output_pointer);
    EXPECT_EQ(Peer::output_records[0].pointer, Peer::output_records[0].ort_pointer);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
    EXPECT_TRUE(Peer::input_owner.expired());
    EXPECT_EQ(executor.successful_runs(), 1U);
  }
  EXPECT_TRUE(Peer::session_owner.expired());
  transport_.reset();
  node_.reset();
  ExpectPayload(*output);
}

TEST_F(NativeExecutorTest, DynamicFallbackExplicitlyCopiesActualShapeAndPayload)
{
  std::optional<native::TensorList> output;
  {
    inference::NativeOnnxExecutor executor(Config(false));
    output.emplace(executor.Run(Input(), *transport_));
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_FALSE(Peer::output_records[0].pointer_identity);
    EXPECT_NE(Peer::output_records[0].pointer, Peer::output_records[0].ort_pointer);
    EXPECT_EQ(Peer::output_records[0].lifetime_path,
      "generic dynamic output explicitly copied D2D into native CUDA Buffer");
    EXPECT_TRUE(Peer::input_owner.expired());
  }
  EXPECT_TRUE(Peer::session_owner.expired());
  transport_.reset();
  node_.reset();
  ExpectPayload(*output);
}

TEST_F(NativeExecutorTest, OutputContractsFailRatherThanSilentlyFallingBack)
{
  for (const auto & contract : {"result=int64[1,4]", "missing=float32[1,4]",
      "result=float32[0,4]", "result=float32[1,9223372036854775807]"})
  {
    auto config = Config();
    config.output_contracts = {contract};
    EXPECT_ANY_THROW(inference::NativeOnnxExecutor executor(config));
  }
  auto config = Config();
  config.output_contracts = {"result=float32[1,4]", "result=float32[1,4]"};
  EXPECT_THROW(inference::NativeOnnxExecutor executor(config), std::invalid_argument);
  config.output_contracts = {"result=float32[1,5]"};
  inference::NativeOnnxExecutor executor(config);
  EXPECT_ANY_THROW(executor.Run(Input(), *transport_));
  EXPECT_FALSE(executor.poisoned());
}

TEST_F(NativeExecutorTest, InputNameRankAndDtypeAreValidatedBeforeSubmission)
{
  inference::NativeOnnxExecutor executor(Config());
  EXPECT_THROW(executor.Run(Input("wrong"), *transport_), std::invalid_argument);
  EXPECT_THROW(executor.Run(Input("values", {4}), *transport_), std::invalid_argument);
  EXPECT_THROW(executor.Run(Input("values", {2, 2}), *transport_), std::invalid_argument);
  auto values = std::make_shared<std::vector<int64_t>>(4, 2);
  auto wrong_type = transport_->Allocate({}, {{"values", 0, 64, 1, {1, 4}}});
  wrong_type.CopyFrom({{values->data(), 4 * sizeof(int64_t), false}}, values);
  EXPECT_THROW(executor.Run(wrong_type.message(), *transport_), std::invalid_argument);
  auto unsupported_type = transport_->Allocate({}, {{"values", 0, 32, 1, {1, 4}}});
  unsupported_type.CopyFrom({{values->data(), 4 * sizeof(int32_t), false}}, values);
  EXPECT_THROW(executor.Run(unsupported_type.message(), *transport_), std::invalid_argument);
  auto extra = transport_->Allocate({},
    {{"values", 0, 64, 1, {1, 4}}, {"extra", 0, 64, 1, {1, 4}}});
  extra.CopyFrom({{values->data(), 4 * sizeof(int64_t), false},
    {values->data(), 4 * sizeof(int64_t), false}}, values);
  EXPECT_THROW(executor.Run(extra.message(), *transport_), std::invalid_argument);
  EXPECT_EQ(executor.successful_runs(), 0U);
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_FALSE(executor.poisoned());
  ExpectPayload(executor.Run(Input(), *transport_));
}

TEST_F(NativeExecutorTest, FailureBeforeSubmissionReclaimsOwnersAndAllowsNextRun)
{
  inference::NativeOnnxExecutor executor(Config());
  Peer::point = Peer::Point::kBeforeRun;
  EXPECT_THROW(executor.Run(Input(), *transport_), std::runtime_error);
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_TRUE(Peer::output_owner.expired());
  EXPECT_FALSE(executor.poisoned());
  ExpectPayload(executor.Run(Input(), *transport_));
}

TEST_F(NativeExecutorTest, SynchronizedFailureReclaimsOwnersWithoutQuarantine)
{
  inference::NativeOnnxExecutor executor(Config());
  Peer::point = Peer::Point::kSynchronized;
  EXPECT_THROW(executor.Run(Input(), *transport_), std::runtime_error);
  EXPECT_TRUE(Peer::input_owner.expired());
  EXPECT_TRUE(Peer::output_owner.expired());
  EXPECT_FALSE(executor.poisoned());
  EXPECT_EQ(executor.successful_runs(), 1U);
  ExpectPayload(executor.Run(Input(), *transport_));
}

TEST_F(NativeExecutorTest, ReportFailureDoesNotQuarantineCompletedGpuWork)
{
  auto config = Config();
  config.binding_report_path = "/dev/null/native-binding-report.json";
  inference::NativeOnnxExecutor executor(config);
  for (size_t i = 1; i <= 2; ++i) {
    EXPECT_ANY_THROW(executor.Run(Input(), *transport_));
    EXPECT_FALSE(executor.poisoned());
    EXPECT_TRUE(Peer::input_owner.expired());
    EXPECT_TRUE(Peer::output_owner.expired());
    EXPECT_EQ(executor.successful_runs(), i);
  }
}

TEST_F(NativeExecutorTest, SubmittedFailureRetainsOwnersUntilRealCudaCompletion)
{
  inference::NativeOnnxExecutor executor(Config());
  PendingGpuWork work;
  std::promise<void> submitted;
  auto submitted_future = submitted.get_future();
  std::weak_ptr<const void> input_owner;
  std::weak_ptr<const void> output_owner;
  auto run = std::async(std::launch::async,
    [&, input = std::optional<native::TensorList>(Input())]() mutable {
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
      (void)executor.Run(*input, *transport_);
      input.reset();
      if (!signalled) {
        submitted.set_exception(std::make_exception_ptr(
          std::runtime_error("native submission hook did not execute")));
      }
      return false;
    } catch (...) {
      input.reset();
      if (!signalled) {
        submitted.set_exception(std::current_exception());
      }
      try {
        throw;
      } catch (const std::runtime_error & error) {
        return std::string(error.what()) == "injected native executor failure";
      } catch (...) {
        return false;
      }
    }
  });
  submitted_future.get();
  work.WaitUntilEntered();
  EXPECT_EQ(work.Query(), cudaErrorNotReady);
  EXPECT_FALSE(input_owner.expired());
  EXPECT_FALSE(output_owner.expired());
  work.Release();
  EXPECT_TRUE(run.get());
  work.Finish();
  EXPECT_TRUE(input_owner.expired());
  EXPECT_TRUE(output_owner.expired());
  EXPECT_FALSE(executor.poisoned());
  ExpectPayload(executor.Run(Input(), *transport_));
}

TEST_F(NativeExecutorTest, UnknownCompletionRetainsFullRequestAndPoisonsExecutor)
{
  auto queued_input = Input();
  PendingGpuWork work;
  std::weak_ptr<inference::OnnxSession> session;
  std::weak_ptr<const void> input;
  std::weak_ptr<const void> output;
  bool submitted = false;
  {
    inference::NativeOnnxExecutor executor(Config());
    Peer::point = Peer::Point::kSubmissionBoundary;
    // This injects an unconfirmable-completion decision, not a real driver fault.
    // The work whose owners are observed is nevertheless real pending CUDA work.
    Peer::unknown_completion = true;
    Peer::submission_work = [&](void * pointer) {
      work.Submit(pointer);
      submitted = true;
    };
    EXPECT_THROW(executor.Run(Input(), *transport_), std::runtime_error);
    ASSERT_TRUE(submitted);
    work.WaitUntilEntered();
    EXPECT_EQ(work.Query(), cudaErrorNotReady);
    input = Peer::input_owner;
    output = Peer::output_owner;
    session = Peer::session_owner;
    EXPECT_FALSE(input.expired());
    EXPECT_FALSE(output.expired());
    EXPECT_TRUE(executor.poisoned());
    EXPECT_THROW(executor.Run(queued_input, *transport_), std::runtime_error);
  }
  EXPECT_FALSE(session.expired());
  work.Release();
  work.Finish();
  EXPECT_FALSE(input.expired());
  EXPECT_FALSE(output.expired());
  EXPECT_FALSE(session.expired());
}

TEST_F(NativeExecutorTest, ShutdownDrainsActiveCallbackAndRejectsQueuedCallback)
{
  auto inference_node = std::make_unique<inference::NativeOnnxInferenceNode>(rclcpp::NodeOptions{});
  auto callback = Peer::Callback(*inference_node);
  auto state = Peer::State(*inference_node);
  auto input = Input();
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  std::atomic<size_t> calls{0};
  Peer::callback_entered = [&] {
    ++calls;
    entered.set_value();
    release_future.wait();
  };
  auto active = std::async(std::launch::async, [&] { callback(input); });
  entered_future.wait();
  auto shutdown = std::async(std::launch::async, [&] { inference_node.reset(); });
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [&] { return state->shutting_down; });
    EXPECT_EQ(state->active_callbacks, 1U);
    EXPECT_EQ(state->node, nullptr);
  }
  callback(input);
  EXPECT_EQ(calls.load(), 1U);
  release.set_value();
  active.get();
  shutdown.get();
  callback(input);
  EXPECT_EQ(calls.load(), 1U);
  EXPECT_EQ(state->active_callbacks, 0U);
}

TEST_F(NativeExecutorTest, NativeNodeRejectsUnsupportedParameters)
{
  for (const auto & parameter : {
      rclcpp::Parameter("execution_provider", "cpu"),
      rclcpp::Parameter("transport", "managed"),
      rclcpp::Parameter("gpu_device_id", -1),
      rclcpp::Parameter("ort_profile_frames", -1),
      rclcpp::Parameter("ort_profile_frames", 1)})
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({parameter});
    EXPECT_THROW(inference::NativeOnnxInferenceNode node(options), std::invalid_argument);
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("model_file_path", "/missing/native-model.onnx")});
  EXPECT_ANY_THROW(inference::NativeOnnxInferenceNode node(options));
}

TEST_F(NativeExecutorTest, RealYoloUsesFormalDirectOutputContract)
{
  auto config = Config();
  config.session.model_file_path = TEST_CUDA_IO_BINDING_MODEL_PATH;
  config.output_contracts = {"output0=float32[1,84,8400]"};
  auto zeros = std::make_shared<std::vector<float>>(3U * 640U * 640U, 0.0F);
  auto batch = transport_->Allocate({}, {{"images", 2, 32, 1, {1, 3, 640, 640}}});
  batch.CopyFrom({{zeros->data(), zeros->size() * sizeof(float), false}}, zeros);
  std::optional<native::TensorList> output;
  {
    inference::NativeOnnxExecutor executor(config);
    output.emplace(executor.Run(batch.message(), *transport_));
    ASSERT_EQ(Peer::output_records.size(), 1U);
    EXPECT_TRUE(Peer::output_records[0].pointer_identity);
    EXPECT_EQ(Peer::output_records[0].pointer, Peer::output_records[0].ort_pointer);
    EXPECT_EQ(Peer::output_pointer, output->data(0));
  }
  ASSERT_EQ(output->tensors().size(), 1U);
  EXPECT_EQ(output->tensors()[0].name, "output0");
  EXPECT_EQ(output->tensors()[0].shape, (std::vector<int64_t>{1, 84, 8400}));
  transport_.reset();
  node_.reset();
  std::vector<float> values(84U * 8400U);
  ASSERT_EQ(cudaMemcpy(values.data(), output->data(0), values.size() * sizeof(float),
      cudaMemcpyDeviceToHost), cudaSuccess);
  for (float value : values) {
    ASSERT_TRUE(std::isfinite(value));
  }
}

} // namespace
