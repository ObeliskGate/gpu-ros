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

#ifndef GPU_ROS_ONNX_INFERENCE_TEST__CANCELLATION_TEST_PEER_HPP_
#define GPU_ROS_ONNX_INFERENCE_TEST__CANCELLATION_TEST_PEER_HPP_

#include <memory>
#include <stdexcept>
#include <vector>

#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"

namespace gpu_ros::onnx_inference
{

// Included only by the test executable and its compilation of the real core.
class OnnxInferenceCancellationTestPeer
{
public:
  enum class Point
  {
    kNone,
    kFirstWriter,
    kSecondBind,
    kSubmissionBoundary
  };

  static void Arm(Point point)
  {
    first_buffer.reset();
    second_buffer.reset();
    retained_owner.reset();
    second_retained_owner.reset();
    armed = point;
    fired = false;
  }

  static void AfterFirstWriter(const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer)
  {
    first_buffer = buffer;
    Inject(Point::kFirstWriter);
  }

  static void AfterFirstWriter(const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer,
    gpu_ros_managed::WriteHandle & writer)
  {
    // A failed/orphaned producer retains this owner along with its allocation;
    // observing the DeviceBuffer wrapper alone would miss that leak.
    auto owner = std::make_shared<int>(0);
    retained_owner = owner;
    writer.retain_owner(std::move(owner));
    AfterFirstWriter(buffer);
  }

  static void BeforeSecondBind(const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer)
  {
    second_buffer = buffer;
    Inject(Point::kSecondBind);
  }

  static void BeforeSecondBind(const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer,
    gpu_ros_managed::WriteHandle & writer)
  {
    auto owner = std::make_shared<int>(0);
    second_retained_owner = owner;
    writer.retain_owner(std::move(owner));
    BeforeSecondBind(buffer);
  }

  static void AfterSubmissionBoundary() { Inject(Point::kSubmissionBoundary); }

  static std::vector<size_t> Available(const OnnxInferenceCore & core)
  {
    std::vector<size_t> available;
    for (const auto & pool : core.managed_output_pools_) {
      available.push_back(pool->available());
    }
    return available;
  }

  inline static thread_local std::shared_ptr<gpu_ros_managed::DeviceBuffer> first_buffer;
  inline static thread_local std::shared_ptr<gpu_ros_managed::DeviceBuffer> second_buffer;
  inline static thread_local std::weak_ptr<const void> retained_owner;
  inline static thread_local std::weak_ptr<const void> second_retained_owner;
  inline static thread_local bool fired{false};

private:
  inline static thread_local Point armed{Point::kNone};

  static void Inject(Point point)
  {
    if (armed == point) {
      armed = Point::kNone;
      fired = true;
      throw std::runtime_error("injected managed output cancellation failure");
    }
  }
};

} // namespace gpu_ros::onnx_inference

#endif // GPU_ROS_ONNX_INFERENCE_TEST__CANCELLATION_TEST_PEER_HPP_
