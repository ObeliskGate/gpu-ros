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

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "gpu_ros_managed_hip/hip_backend.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "gpu_ros_onnx_inference/staging_pool_limits.hpp"

namespace gpu_ros::onnx_inference
{
using TensorBundleMsg = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

namespace
{
int DeclareNonnegativeDeviceId(rclcpp::Node * node)
{
  const int value = node->declare_parameter<int>("gpu_device_id", 0);
  if (value < 0) {
    throw std::invalid_argument("gpu_device_id must be non-negative");
  }
  return value;
}

size_t DeclarePositiveSize(rclcpp::Node * node, const char * name, int64_t default_value)
{
  const int64_t value = node->declare_parameter<int64_t>(name, default_value);
  if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(name) + " must be positive and fit in size_t");
  }
  return static_cast<size_t>(value);
}

std::chrono::milliseconds DeclareNonnegativeTimeout(
  rclcpp::Node * node, const char * name, int64_t default_value)
{
  const int64_t value = node->declare_parameter<int64_t>(name, default_value);
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
  return std::chrono::milliseconds(value);
}

} // namespace

void CheckHipDevice(const gpu_ros_managed::ManagedTensor & tensor, int gpu_device_id)
{
  const auto * buffer =
    std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&tensor.storage());
  if (buffer == nullptr || !*buffer) {
    throw std::invalid_argument(
      "Managed HIP TensorBundle tensor '" + tensor.name() + "' is not device-backed");
  }
  if ((*buffer)->device_id() !=
      gpu_ros_managed::DeviceId{gpu_ros_managed::BackendKind::kHip, gpu_device_id})
  {
    throw std::invalid_argument(
      "Managed HIP TensorBundle tensor '" + tensor.name() + "' is on the wrong backend or device");
  }
}

class StdToManagedHipTensorBundleNode final : public rclcpp::Node
{
public:
  explicit StdToManagedHipTensorBundleNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
      : rclcpp::Node("std_to_managed_hip_tensor_bundle_node", options),
        gpu_device_id_(DeclareNonnegativeDeviceId(this)),
        pool_capacity_(DeclarePositiveSize(this, "managed_pool_capacity", 16)),
        pool_timeout_(DeclareNonnegativeTimeout(this, "managed_pool_wait_timeout_ms", 100)),
        max_tensor_bytes_(DeclarePositiveSize(this, "managed_max_tensor_bytes", 67108864)),
        pool_cache_max_bytes_(
          DeclarePositiveSize(this, "managed_pool_cache_max_bytes", 1073741824)),
        pool_cache_max_entries_(DeclarePositiveSize(this, "managed_pool_cache_max_entries", 16)),
        stream_(gpu_ros_managed::hip::make_stream(gpu_device_id_)),
        publisher_(this, "tensor_output", rclcpp::QoS(10))
  {
    subscription_ = create_subscription<TensorBundleMsg>("tensor_input", rclcpp::QoS(10),
      [this](const TensorBundleMsg::SharedPtr message) { OnTensorBundle(message); });
  }

  ~StdToManagedHipTensorBundleNode() override
  {
    subscription_.reset();
    publisher_.reset();
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (const auto & entry : pools_) {
      if (entry.second && !entry.second->shutdown(std::chrono::seconds(5))) {
        RCLCPP_ERROR(
          get_logger(), "Std-to-Managed HIP pool did not drain for %zu-byte blocks", entry.first);
      }
    }
  }

private:
  std::shared_ptr<gpu_ros_managed::FixedDeviceMemoryPool> GetPool(size_t bytes)
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    const auto found = pools_.find(bytes);
    if (found != pools_.end()) {
      return found->second;
    }
    const size_t next_cache_bytes =
      detail::ValidateStagingPoolAddition(bytes, pool_capacity_, pool_cache_bytes_, pools_.size(),
        max_tensor_bytes_, pool_cache_max_bytes_, pool_cache_max_entries_);
    auto pool = std::make_shared<gpu_ros_managed::FixedDeviceMemoryPool>(
      gpu_ros_managed::hip::make_fixed_device_pool(
        bytes, static_cast<size_t>(pool_capacity_), gpu_device_id_));
    pools_.emplace(bytes, pool);
    pool_cache_bytes_ = next_cache_bytes;
    return pool;
  }

  void OnTensorBundle(const TensorBundleMsg::SharedPtr & message)
  {
    std::vector<std::unique_ptr<gpu_ros_managed::PoolBlock>> blocks;
    std::vector<bool> copy_submitted;
    std::vector<bool> writer_active;
    try {
      const auto selected = hipSetDevice(gpu_device_id_);
      if (selected != hipSuccess) {
        throw std::runtime_error(
          "Std-to-Managed HIP hipSetDevice failed: " + std::string(hipGetErrorName(selected)) +
          " (" + hipGetErrorString(selected) + ")");
      }
      blocks.reserve(message->tensors.size());
      copy_submitted.reserve(message->tensors.size());
      writer_active.reserve(message->tensors.size());
      std::vector<gpu_ros_managed::ManagedTensor> tensors;
      tensors.reserve(message->tensors.size());

      const auto reservation_deadline = std::chrono::steady_clock::now() + pool_timeout_;
      const auto remaining_timeout = [&reservation_deadline]() {
        const auto now = std::chrono::steady_clock::now();
        return now >= reservation_deadline ? std::chrono::milliseconds(0)
                                           : std::chrono::duration_cast<std::chrono::milliseconds>(
                                               reservation_deadline - now);
      };

      for (const auto & tensor : message->tensors) {
        const std::vector<int64_t> shape(tensor.shape.begin(), tensor.shape.end());
        const auto dtype = static_cast<gpu_ros_managed::TensorDataType>(tensor.data_type);
        const size_t bytes = gpu_ros_managed::tensor_byte_size(shape, dtype);
        if (tensor.data.size() < bytes) {
          throw std::invalid_argument(
            "Standard TensorBundle tensor '" + tensor.name + "' has insufficient data");
        }

        auto block = GetPool(bytes)->acquire_for(stream_.stream(), remaining_timeout());
        if (!block) {
          ++pool_exhaustion_drops_;
          throw std::runtime_error("Std-to-Managed HIP pool exhausted for '" + tensor.name + "'");
        }
        block->writer.retain_owner(std::static_pointer_cast<const void>(message));
        blocks.push_back(std::move(block));
        copy_submitted.push_back(false);
        writer_active.push_back(true);
      }

      for (size_t index = 0; index < message->tensors.size(); ++index) {
        const auto & tensor = message->tensors[index];
        const size_t bytes = blocks[index]->buffer->size();
        copy_submitted[index] = true;
        const auto result = hipMemcpyAsync(blocks[index]->writer.data(), tensor.data.data(), bytes,
          hipMemcpyHostToDevice, stream_.get());
        if (result != hipSuccess) {
          throw std::runtime_error(
            "Std-to-Managed HIP H2D copy failed: " + std::string(hipGetErrorName(result)) + " (" +
            hipGetErrorString(result) + ")");
        }
        blocks[index]->writer.finalize();
        writer_active[index] = false;
        const auto & source = message->tensors[index];
        const auto shape = std::vector<int64_t>(source.shape.begin(), source.shape.end());
        tensors.emplace_back(source.name,
          static_cast<gpu_ros_managed::TensorDataType>(source.data_type), shape,
          blocks[index]->buffer);
      }
      publisher_.publish(gpu_ros_managed::ManagedTensorBundle(message->header, std::move(tensors)));
    } catch (const std::exception & error) {
      for (size_t index = 0; index < blocks.size(); ++index) {
        if (!writer_active[index]) {
          continue;
        }
        if (copy_submitted[index]) {
          blocks[index]->writer.fail();
        } else {
          blocks[index]->writer.cancel();
        }
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Failed to stage standard TensorBundle to HIP (%zu pool drops): %s",
        pool_exhaustion_drops_.load(), error.what());
    } catch (...) {
      // WriteHandle destruction marks every still-active reservation failed.
      // That #3 invariant makes the callback's #11-style exception boundary
      // safe even when the exception type is unknown.
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Dropping standard TensorBundle after unknown HIP staging failure");
    }
  }

  int gpu_device_id_;
  size_t pool_capacity_;
  std::chrono::milliseconds pool_timeout_;
  size_t max_tensor_bytes_;
  size_t pool_cache_max_bytes_;
  size_t pool_cache_max_entries_;
  gpu_ros_managed::hip::HipStream stream_;
  gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle> publisher_;
  rclcpp::Subscription<TensorBundleMsg>::SharedPtr subscription_;
  std::mutex pool_mutex_;
  std::unordered_map<size_t, std::shared_ptr<gpu_ros_managed::FixedDeviceMemoryPool>> pools_;
  size_t pool_cache_bytes_{0};
  std::atomic<size_t> pool_exhaustion_drops_{0};
};

class ManagedHipToStdTensorBundleNode final : public rclcpp::Node
{
public:
  explicit ManagedHipToStdTensorBundleNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
      : rclcpp::Node("managed_hip_to_std_tensor_bundle_node", options),
        gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
        publisher_(create_publisher<TensorBundleMsg>("tensor_output", rclcpp::QoS(10)))
  {
    subscription_ = std::make_unique<
      gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>(
      this, "tensor_input",
      [this](gpu_ros_managed::ManagedTensorBundleView input) { OnTensorBundle(std::move(input)); },
      rclcpp::QoS(10));
  }

private:
  void OnTensorBundle(gpu_ros_managed::ManagedTensorBundleView input)
  {
    try {
      const auto selected = hipSetDevice(gpu_device_id_);
      if (selected != hipSuccess) {
        throw std::runtime_error(
          "Managed-to-standard HIP hipSetDevice failed: " + std::string(hipGetErrorName(selected)) +
          " (" + hipGetErrorString(selected) + ")");
      }
      auto output = std::make_unique<TensorBundleMsg>();
      output->header = input.header();
      output->tensors.reserve(input.tensors().size());
      for (const auto & tensor : input.tensors()) {
        CheckHipDevice(tensor, gpu_device_id_);
        const auto & buffer =
          std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
        auto & message_tensor = output->tensors.emplace_back();
        message_tensor.name = tensor.name();
        message_tensor.data_type = static_cast<uint8_t>(tensor.data_type());
        message_tensor.shape.assign(tensor.shape().begin(), tensor.shape().end());
        message_tensor.data.resize(tensor.byte_size());
        buffer->copy_to_host_blocking(message_tensor.data.data(), message_tensor.data.size());
      }
      publisher_->publish(std::move(output));
    } catch (const std::exception & error) {
      RCLCPP_ERROR(
        get_logger(), "Failed to stage Managed HIP TensorBundle to host: %s", error.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
        "Dropping Managed HIP TensorBundle after unknown staging failure");
    }
  }

  int gpu_device_id_;
  rclcpp::Publisher<TensorBundleMsg>::SharedPtr publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>
    subscription_;
};
} // namespace gpu_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::StdToManagedHipTensorBundleNode)
RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::onnx_inference::ManagedHipToStdTensorBundleNode)
