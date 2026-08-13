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

#include "isaac_ros_rtdetr_std/rtdetr_managed_hip_nodes.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "isaac_ros_detection_common/hip_preprocess.hpp"
#include "isaac_ros_detection_common/image_preprocess.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia::isaac_ros::rtdetr_std
{
namespace
{
constexpr size_t kChannels = 3U;

void CheckHip(hipError_t error, const char * operation)
{
  if (error == hipSuccess) {return;}
  throw std::runtime_error(
          std::string(operation) + ": " + hipGetErrorName(error) +
          " (" + hipGetErrorString(error) + ")");
}

size_t CheckedProduct(size_t first, size_t second, const char * name)
{
  if (first != 0U && second > std::numeric_limits<size_t>::max() / first) {
    throw std::overflow_error(std::string(name) + " overflows size_t");
  }
  return first * second;
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
  if (value < 0) {throw std::invalid_argument(std::string(name) + " must be non-negative");}
  return std::chrono::milliseconds(value);
}

std::shared_ptr<gpu_ros_managed::DeviceBuffer> RequireDeviceTensor(
  const gpu_ros_managed::ManagedTensor & tensor,
  const std::string & name,
  gpu_ros_managed::TensorDataType dtype,
  int device_id)
{
  if (tensor.is_host() || tensor.data_type() != dtype) {
    throw std::invalid_argument("RT-DETR Managed tensor '" + name + "' has wrong storage/dtype");
  }
  const auto & buffer = std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
    tensor.storage());
  if (!buffer || buffer->device_id() != gpu_ros_managed::DeviceId{
        gpu_ros_managed::BackendKind::kHip, device_id})
  {
    throw std::invalid_argument("RT-DETR Managed tensor '" + name + "' has the wrong HIP device");
  }
  if (buffer->readiness() == gpu_ros_managed::BufferReadiness::kNotReady) {
    throw std::invalid_argument("RT-DETR Managed tensor '" + name + "' is not ready");
  }
  return buffer;
}

template<typename T>
void ValidateShape(
  const gpu_ros_managed::ManagedTensor & tensor,
  const std::vector<int64_t> & expected, const std::string & name)
{
  if (tensor.shape() != expected || tensor.byte_size() !=
    gpu_ros_managed::tensor_byte_size(expected,
      std::is_same_v<T, float> ? gpu_ros_managed::TensorDataType::kFloat32 :
      gpu_ros_managed::TensorDataType::kInt64))
  {
    throw std::invalid_argument("RT-DETR Managed tensor '" + name + "' has the wrong shape");
  }
}
}  // namespace

RtDetrManagedHipImageEncoderNode::RtDetrManagedHipImageEncoderNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_managed_hip_image_encoder_node", options),
  tensor_name_(declare_parameter<std::string>("tensor_name", "input_tensor")),
  output_width_(declare_parameter<int64_t>("output_width", 640)),
  output_height_(declare_parameter<int64_t>("output_height", 640)),
  gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
  pool_capacity_(DeclarePositiveSize(this, "managed_pool_capacity", 16)),
  pool_timeout_(DeclareNonnegativeTimeout(this, "managed_pool_wait_timeout_ms", 100)),
  max_input_bytes_(DeclarePositiveSize(this, "managed_max_input_bytes", 16 * 1024 * 1024)),
  publisher_(this, "managed_tensor_output", rclcpp::QoS(10))
{
  if (tensor_name_.empty() || output_width_ <= 0 || output_height_ <= 0 ||
    output_width_ > std::numeric_limits<int>::max() ||
    output_height_ > std::numeric_limits<int>::max() || gpu_device_id_ < 0 ||
    pool_capacity_ == 0U || pool_timeout_.count() < 0 || max_input_bytes_ == 0U)
  {
    throw std::invalid_argument("invalid RT-DETR Managed HIP encoder parameters");
  }
  const size_t output_bytes = CheckedProduct(
    CheckedProduct(kChannels, static_cast<size_t>(output_width_), "output tensor"),
    CheckedProduct(static_cast<size_t>(output_height_), sizeof(float), "output tensor"),
    "output tensor");
  stream_ = std::make_unique<gpu_ros_managed::hip::HipStream>(
    gpu_ros_managed::hip::make_stream(gpu_device_id_));
  raw_pool_ = std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
    gpu_ros_managed::hip::make_fixed_device_pool(max_input_bytes_, pool_capacity_, gpu_device_id_));
  output_pool_ = std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
    gpu_ros_managed::hip::make_fixed_device_pool(output_bytes, pool_capacity_, gpu_device_id_));
  subscription_ = create_subscription<sensor_msgs::msg::Image>(
    "image", 10,
    std::bind(&RtDetrManagedHipImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

RtDetrManagedHipImageEncoderNode::~RtDetrManagedHipImageEncoderNode()
{
  subscription_.reset();
  publisher_.reset();
  if (output_pool_ && !output_pool_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "RT-DETR Managed output pool did not drain during shutdown");
  }
  if (raw_pool_ && !raw_pool_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "RT-DETR Managed raw-image pool did not drain during shutdown");
  }
}

void RtDetrManagedHipImageEncoderNode::InputCallback(
  sensor_msgs::msg::Image::ConstSharedPtr message)
{
  std::unique_ptr<gpu_ros_managed::PoolBlock> output;
  std::unique_ptr<gpu_ros_managed::PoolBlock> raw;
  bool raw_work_submitted = false;
  bool output_work_submitted = false;
  try {
    const auto plan = detection_common::MakeImagePreprocessPlan(
      *message, output_width_, output_height_,
      detection_common::PreprocessNormalization::kNone);
    const size_t source_bytes = CheckedProduct(
      static_cast<size_t>(plan.input_step), static_cast<size_t>(plan.input_height),
      "source image");
    const size_t copy_width = detection_common::SourceRowBytes(plan);
    if (source_bytes > max_input_bytes_) {
      throw std::runtime_error("source image exceeds managed_max_input_bytes");
    }
    const auto reservation_deadline = std::chrono::steady_clock::now() + pool_timeout_;
    const auto remaining_timeout = [&reservation_deadline]() {
        const auto now = std::chrono::steady_clock::now();
        return now >= reservation_deadline ? std::chrono::milliseconds(0) :
               std::chrono::duration_cast<std::chrono::milliseconds>(reservation_deadline - now);
      };
    output = output_pool_->acquire_for(stream_->stream(), remaining_timeout());
    if (!output) {
      ++pool_exhaustion_drops_;
      throw std::runtime_error("RT-DETR Managed output pool exhausted");
    }
    raw = raw_pool_->acquire_for(stream_->stream(), remaining_timeout());
    if (!raw) {
      ++pool_exhaustion_drops_;
      throw std::runtime_error("RT-DETR Managed raw-image pool exhausted");
    }
    raw->writer.retain_owner(std::static_pointer_cast<const void>(message));
    raw_work_submitted = true;
    CheckHip(hipMemcpy2DAsync(
      raw->writer.data(), plan.input_step, message->data.data(), message->step,
      copy_width, plan.input_height, hipMemcpyHostToDevice, stream_->get()),
      "RT-DETR Managed H2D image copy");
    output_work_submitted = true;
    detection_common::LaunchHipPreprocess(
      raw->writer.data(), reinterpret_cast<float *>(output->writer.data()), plan, stream_->get());

    gpu_ros_managed::ManagedTensor tensor(
      tensor_name_, gpu_ros_managed::TensorDataType::kFloat32,
      {1, static_cast<int64_t>(kChannels), output_height_, output_width_}, output->buffer);
    output->writer.finalize();
    publisher_.publish(gpu_ros_managed::ManagedTensorList(
        message->header, std::vector<gpu_ros_managed::ManagedTensor>{std::move(tensor)}));
  } catch (const std::exception & error) {
    if (output) {
      if (output_work_submitted) {output->writer.fail();} else {output->writer.cancel();}
    }
    if (raw) {
      if (raw_work_submitted) {raw->writer.fail();} else {raw->writer.cancel();}
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "RT-DETR Managed HIP encoder dropped frame (%zu pool drops): %s",
      pool_exhaustion_drops_, error.what());
  }
}

RtDetrManagedHipPreprocessorNode::RtDetrManagedHipPreprocessorNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_managed_hip_preprocessor_node", options),
  input_image_tensor_name_(declare_parameter<std::string>(
      "input_image_tensor_name", "input_tensor")),
  output_image_tensor_name_(declare_parameter<std::string>(
      "output_image_tensor_name", "images")),
  output_size_tensor_name_(declare_parameter<std::string>(
      "output_size_tensor_name", "orig_target_sizes")),
  image_width_(declare_parameter<int64_t>("image_width", 640)),
  image_height_(declare_parameter<int64_t>("image_height", 480)),
  model_input_width_(declare_parameter<int64_t>("model_input_width", 640)),
  model_input_height_(declare_parameter<int64_t>("model_input_height", 640)),
  use_max_dim_for_orig_size_(declare_parameter<bool>("use_max_dim_for_orig_size", true)),
  gpu_device_id_(declare_parameter<int>("gpu_device_id", 0)),
  pool_capacity_(DeclarePositiveSize(this, "managed_pool_capacity", 16)),
  pool_timeout_(DeclareNonnegativeTimeout(this, "managed_pool_wait_timeout_ms", 100)),
  stream_(gpu_ros_managed::hip::make_stream(gpu_device_id_)),
  publisher_(this, "managed_tensor_output", rclcpp::QoS(10))
{
  if (input_image_tensor_name_.empty() || output_image_tensor_name_.empty() ||
    output_size_tensor_name_.empty() || image_width_ <= 0 || image_height_ <= 0 ||
    model_input_width_ <= 0 || model_input_height_ <= 0 ||
    model_input_width_ > std::numeric_limits<int>::max() ||
    model_input_height_ > std::numeric_limits<int>::max() || gpu_device_id_ < 0 ||
    pool_capacity_ == 0U || pool_timeout_.count() < 0)
  {
    throw std::invalid_argument("invalid RT-DETR Managed HIP preprocessor parameters");
  }
  size_pool_ = std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
    gpu_ros_managed::hip::make_fixed_device_pool(
      2U * sizeof(int64_t), pool_capacity_, gpu_device_id_));
  subscriber_ = std::make_unique<gpu_ros_managed::ManagedSubscriber<
      gpu_ros_managed::ManagedTensorListView>>(
    this, "managed_tensor_input", std::bind(
      &RtDetrManagedHipPreprocessorNode::InputCallback, this, std::placeholders::_1),
    rclcpp::QoS(10));
}

RtDetrManagedHipPreprocessorNode::~RtDetrManagedHipPreprocessorNode()
{
  subscriber_.reset();
  publisher_.reset();
  if (size_pool_ && !size_pool_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(
      get_logger(), "RT-DETR Managed orig_target_sizes pool did not drain during shutdown");
  }
}

void RtDetrManagedHipPreprocessorNode::InputCallback(
  gpu_ros_managed::ManagedTensorListView message)
{
  std::unique_ptr<gpu_ros_managed::PoolBlock> size_block;
  try {
    const auto & input = message.get_tensor(input_image_tensor_name_);
    const auto input_buffer = RequireDeviceTensor(
      input, input_image_tensor_name_, gpu_ros_managed::TensorDataType::kFloat32,
      gpu_device_id_);
    if (input.shape() != std::vector<int64_t>{
        1, 3, model_input_height_, model_input_width_})
    {
      throw std::invalid_argument(
              "RT-DETR Managed preprocessor input does not match [1,3,model_input_height,"
              "model_input_width]");
    }
    size_block = size_pool_->acquire_for(stream_.stream(), pool_timeout_);
    if (!size_block) {
      ++pool_exhaustion_drops_;
      throw std::runtime_error("RT-DETR Managed orig_target_sizes pool exhausted");
    }
    const int64_t original_width = use_max_dim_for_orig_size_ ?
      std::max(image_width_, image_height_) : image_width_;
    const int64_t original_height = use_max_dim_for_orig_size_ ?
      std::max(image_width_, image_height_) : image_height_;
    detection_common::LaunchHipWriteInt64Pair(
      reinterpret_cast<int64_t *>(size_block->writer.data()),
      original_width, original_height, stream_.get());
    gpu_ros_managed::ManagedTensor image_out(
      output_image_tensor_name_, gpu_ros_managed::TensorDataType::kFloat32,
      input.shape(), input_buffer);
    gpu_ros_managed::ManagedTensor size_out(
      output_size_tensor_name_, gpu_ros_managed::TensorDataType::kInt64,
      {1, 2}, size_block->buffer);
    size_block->writer.finalize();
    publisher_.publish(gpu_ros_managed::ManagedTensorList(
        message.get().header(), std::vector<gpu_ros_managed::ManagedTensor>{
        std::move(image_out), std::move(size_out)}));
  } catch (const std::exception & error) {
    if (size_block) {size_block->writer.fail();}
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "RT-DETR Managed HIP preprocessor dropped frame (%zu pool drops): %s",
      pool_exhaustion_drops_, error.what());
  }
}

RtDetrManagedHipDecoderNode::RtDetrManagedHipDecoderNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_managed_hip_decoder_node", options),
  read_stream_(gpu_ros_managed::hip::make_stream(declare_parameter<int>("gpu_device_id", 0)))
{
  config_.labels_tensor_name = declare_parameter<std::string>("labels_tensor_name", "labels");
  config_.boxes_tensor_name = declare_parameter<std::string>("boxes_tensor_name", "boxes");
  config_.scores_tensor_name = declare_parameter<std::string>("scores_tensor_name", "scores");
  config_.confidence_threshold = declare_parameter<double>("confidence_threshold", 0.9);
  publisher_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", 10);
  subscriber_ = std::make_unique<gpu_ros_managed::ManagedSubscriber<
      gpu_ros_managed::ManagedTensorListView>>(
    this, "managed_tensor_input", std::bind(
      &RtDetrManagedHipDecoderNode::InputCallback, this, std::placeholders::_1),
    rclcpp::QoS(10));
}

void RtDetrManagedHipDecoderNode::InputCallback(
  gpu_ros_managed::ManagedTensorListView message)
{
  try {
    const auto & labels = message.get_tensor(config_.labels_tensor_name);
    const auto & boxes = message.get_tensor(config_.boxes_tensor_name);
    const auto & scores = message.get_tensor(config_.scores_tensor_name);
    const auto labels_buffer = RequireDeviceTensor(
      labels, config_.labels_tensor_name, gpu_ros_managed::TensorDataType::kInt64,
      read_stream_.stream().device_id().ordinal);
    const auto boxes_buffer = RequireDeviceTensor(
      boxes, config_.boxes_tensor_name, gpu_ros_managed::TensorDataType::kFloat32,
      read_stream_.stream().device_id().ordinal);
    const auto scores_buffer = RequireDeviceTensor(
      scores, config_.scores_tensor_name, gpu_ros_managed::TensorDataType::kFloat32,
      read_stream_.stream().device_id().ordinal);
    ValidateShape<int64_t>(labels, {1, 300}, config_.labels_tensor_name);
    ValidateShape<float>(boxes, {1, 300, 4}, config_.boxes_tensor_name);
    ValidateShape<float>(scores, {1, 300}, config_.scores_tensor_name);
    std::vector<int64_t> labels_host(300U);
    std::vector<float> boxes_host(1200U);
    std::vector<float> scores_host(300U);
    auto labels_read = labels_buffer->get_read_handle(read_stream_.stream());
    auto boxes_read = boxes_buffer->get_read_handle(read_stream_.stream());
    auto scores_read = scores_buffer->get_read_handle(read_stream_.stream());
    CheckHip(hipMemcpyAsync(
        labels_host.data(), labels_read.data(), labels.byte_size(),
        hipMemcpyDeviceToHost, read_stream_.get()), "RT-DETR Managed labels D2H");
    CheckHip(hipMemcpyAsync(
        boxes_host.data(), boxes_read.data(), boxes.byte_size(),
        hipMemcpyDeviceToHost, read_stream_.get()), "RT-DETR Managed boxes D2H");
    CheckHip(hipMemcpyAsync(
        scores_host.data(), scores_read.data(), scores.byte_size(),
        hipMemcpyDeviceToHost, read_stream_.get()), "RT-DETR Managed scores D2H");
    CheckHip(hipStreamSynchronize(read_stream_.get()), "RT-DETR Managed decoder read sync");
    labels_read.finish();
    boxes_read.finish();
    scores_read.finish();
    publisher_->publish(DecodeRtDetrValues(
      message.header(), labels_host.data(), labels_host.size(), boxes_host.data(),
      boxes_host.size(), scores_host.data(), scores_host.size(), config_));
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "RT-DETR Managed HIP decoder dropped frame: %s", error.what());
  }
}

}  // namespace nvidia::isaac_ros::rtdetr_std

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::rtdetr_std::RtDetrManagedHipImageEncoderNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::rtdetr_std::RtDetrManagedHipPreprocessorNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::rtdetr_std::RtDetrManagedHipDecoderNode)
