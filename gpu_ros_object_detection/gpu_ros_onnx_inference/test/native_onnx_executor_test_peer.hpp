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

#ifndef GPU_ROS_ONNX_INFERENCE_TEST__NATIVE_ONNX_EXECUTOR_TEST_PEER_HPP_
#define GPU_ROS_ONNX_INFERENCE_TEST__NATIVE_ONNX_EXECUTOR_TEST_PEER_HPP_

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "native_onnx_executor.hpp"
#include "gpu_ros_onnx_inference/native_onnx_inference_node.hpp"

namespace gpu_ros::onnx_inference
{

// Included only by the test executable and its compilation of the real sources.
class NativeOnnxExecutorTestPeer
{
public:
  enum class Point { kNone, kBeforeRun, kSubmissionBoundary, kSynchronized };

  static void Reset()
  {
    point = Point::kNone;
    unknown_completion = false;
    submission_work = {};
    input_owner.reset();
    output_owner.reset();
    session_owner.reset();
    output_pointer = nullptr;
    input_records.clear();
    output_records.clear();
  }

  static void BeforeRun(std::shared_ptr<const void> input,
    std::shared_ptr<const void> output, std::shared_ptr<OnnxSession> session, void * pointer)
  {
    input_owner = input;
    output_owner = output;
    session_owner = session;
    output_pointer = pointer;
    Inject(Point::kBeforeRun);
  }
  static void AfterSubmissionBoundary(void *)
  {
    if (submission_work) {
      submission_work(output_pointer);
    }
    Inject(Point::kSubmissionBoundary);
  }
  static void AfterSynchronization() { Inject(Point::kSynchronized); }
  static bool ForceUnknownCompletion() noexcept { return unknown_completion; }
  static void Completed(const std::vector<TensorBindingRecord> & inputs,
    const std::vector<TensorBindingRecord> & outputs)
  {
    input_records = inputs;
    output_records = outputs;
  }

  static auto Callback(const NativeOnnxInferenceNode & node) { return node.MakeCallback(); }
  static auto State(const NativeOnnxInferenceNode & node) { return node.callback_state_; }
  static void CallbackEntered()
  {
    if (callback_entered) {
      callback_entered();
    }
  }

  inline static thread_local Point point{Point::kNone};
  inline static thread_local bool unknown_completion{false};
  inline static thread_local std::function<void(void *)> submission_work;
  inline static thread_local std::weak_ptr<const void> input_owner;
  inline static thread_local std::weak_ptr<const void> output_owner;
  inline static thread_local std::weak_ptr<OnnxSession> session_owner;
  inline static thread_local void * output_pointer{nullptr};
  inline static thread_local std::vector<TensorBindingRecord> input_records;
  inline static thread_local std::vector<TensorBindingRecord> output_records;
  // Set before starting callback threads and cleared only after joining them.
  inline static std::function<void()> callback_entered;

private:
  static void Inject(Point expected)
  {
    if (point == expected) {
      point = Point::kNone;
      throw std::runtime_error("injected native executor failure");
    }
  }
};

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE_TEST__NATIVE_ONNX_EXECUTOR_TEST_PEER_HPP_
