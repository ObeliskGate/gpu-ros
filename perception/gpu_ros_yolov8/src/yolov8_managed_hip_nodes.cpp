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

#include "gpu_ros_yolov8/yolov8_managed_hip_nodes.hpp"

#include <hip/hip_runtime_api.h>

#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_detection_common/hip_preprocess.hpp"
#include "gpu_ros_detection_common/image_preprocess.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::yolov8
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

std::shared_ptr<gpu_ros_managed::DeviceBuffer> DeviceTensorBuffer(
  const gpu_ros_managed::ManagedTensor & tensor, const std::string & name, int device_id)
{
  if (tensor.is_host() || tensor.data_type() != gpu_ros_managed::TensorDataType::kFloat32) {
    throw std::invalid_argument("YOLOv8 Managed tensor '" + name + "' must be HIP float32");
  }
  if (tensor.shape().size() != 3U && tensor.shape().size() != 4U) {
    throw std::invalid_argument("YOLOv8 Managed tensor '" + name + "' has an invalid rank");
  }
  const auto & buffer = std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(
    tensor.storage());
  if (!buffer || buffer->device_id() != gpu_ros_managed::DeviceId{
        gpu_ros_managed::BackendKind::kHip, device_id})
  {
    throw std::invalid_argument("YOLOv8 Managed tensor has the wrong HIP device");
  }
  if (buffer->readiness() == gpu_ros_managed::BufferReadiness::kNotReady) {
    throw std::invalid_argument("YOLOv8 Managed tensor is not ready");
  }
  return buffer;
}
}  // namespace

YoloV8ManagedHipImageEncoderNode::YoloV8ManagedHipImageEncoderNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_managed_hip_image_encoder_node", options),
  tensor_name_(declare_parameter<std::string>("tensor_name", "images")),
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
    throw std::invalid_argument("invalid YOLOv8 Managed HIP encoder parameters");
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
    std::bind(&YoloV8ManagedHipImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

YoloV8ManagedHipImageEncoderNode::~YoloV8ManagedHipImageEncoderNode()
{
  subscription_.reset();
  publisher_.reset();
  if (output_pool_ && !output_pool_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "YOLOv8 Managed output pool did not drain during shutdown");
  }
  if (raw_pool_ && !raw_pool_->shutdown(std::chrono::seconds(5))) {
    RCLCPP_ERROR(get_logger(), "YOLOv8 Managed raw-image pool did not drain during shutdown");
  }
}

void YoloV8ManagedHipImageEncoderNode::InputCallback(
  sensor_msgs::msg::Image::ConstSharedPtr message)
{
  std::unique_ptr<gpu_ros_managed::PoolBlock> output;
  std::unique_ptr<gpu_ros_managed::PoolBlock> raw;
  bool raw_work_submitted = false;
  bool output_work_submitted = false;
  try {
    CheckHip(hipSetDevice(gpu_device_id_), "YOLOv8 Managed encoder hipSetDevice");
    const auto plan = detection_common::MakeImagePreprocessPlan(
      *message, output_width_, output_height_,
      detection_common::PreprocessNormalization::kUnitRange);
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
      throw std::runtime_error("YOLOv8 Managed output pool exhausted");
    }
    raw = raw_pool_->acquire_for(stream_->stream(), remaining_timeout());
    if (!raw) {
      ++pool_exhaustion_drops_;
      throw std::runtime_error("YOLOv8 Managed raw-image pool exhausted");
    }
    raw->writer.retain_owner(std::static_pointer_cast<const void>(message));
    raw_work_submitted = true;
    CheckHip(hipMemcpy2DAsync(
      raw->writer.data(), plan.input_step, message->data.data(), message->step,
      copy_width, plan.input_height, hipMemcpyHostToDevice, stream_->get()),
      "YOLOv8 Managed H2D image copy");
    output_work_submitted = true;
    detection_common::LaunchHipPreprocess(
      raw->writer.data(), reinterpret_cast<float *>(output->writer.data()), plan, stream_->get());
    // The raw image is an internal staging buffer, not a published tensor.
    // Finalize it after queuing the copy and preprocess so the pool can recycle
    // it once this stream reaches the event.
    raw->writer.finalize();

    gpu_ros_managed::ManagedTensor tensor(
      tensor_name_, gpu_ros_managed::TensorDataType::kFloat32,
      {1, static_cast<int64_t>(kChannels), output_height_, output_width_}, output->buffer);
    output->writer.finalize();
    publisher_.publish(gpu_ros_managed::ManagedTensorBundle(
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
      "YOLOv8 Managed HIP encoder dropped frame (%zu pool drops): %s",
      pool_exhaustion_drops_, error.what());
  }
}

YoloV8ManagedHipDecoderNode::YoloV8ManagedHipDecoderNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_managed_hip_decoder_node", options),
  read_stream_(gpu_ros_managed::hip::make_stream(declare_parameter<int>("gpu_device_id", 0)))
{
  config_.tensor_name = declare_parameter<std::string>("tensor_name", "output0");
  config_.confidence_threshold = declare_parameter<double>("confidence_threshold", 0.25);
  config_.nms_threshold = declare_parameter<double>("nms_threshold", 0.45);
  config_.num_classes = declare_parameter<int64_t>("num_classes", 80);
  publisher_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", 10);
  subscriber_ = std::make_unique<gpu_ros_managed::ManagedSubscriber<
        gpu_ros_managed::ManagedTensorBundleView>>(
    this, "managed_tensor_input", std::bind(
      &YoloV8ManagedHipDecoderNode::InputCallback, this, std::placeholders::_1),
    rclcpp::QoS(10));
}

void YoloV8ManagedHipDecoderNode::InputCallback(
  gpu_ros_managed::ManagedTensorBundleView message)
{
  try {
    CheckHip(
      hipSetDevice(read_stream_.stream().device_id().ordinal),
      "YOLOv8 Managed decoder hipSetDevice");
    const auto & tensor = message.get_tensor(config_.tensor_name);
    const auto buffer = DeviceTensorBuffer(tensor, config_.tensor_name,
        read_stream_.stream().device_id().ordinal);
    if (tensor.shape().size() != 3U || tensor.shape()[0] != 1 ||
      tensor.shape()[1] != 4 + config_.num_classes)
    {
      throw std::invalid_argument(
              "YOLOv8 Managed decoder expects output shape [1,4+num_classes,boxes]");
    }
    const size_t count = tensor.byte_size() / sizeof(float);
    std::vector<float> values(count);
    auto read = buffer->get_read_handle(read_stream_.stream());
    CheckHip(hipMemcpyAsync(
        values.data(), read.data(), tensor.byte_size(), hipMemcpyDeviceToHost, read_stream_.get()),
      "YOLOv8 Managed D2H output copy");
    CheckHip(hipStreamSynchronize(read_stream_.get()), "YOLOv8 Managed decoder read sync");
    read.finish();
    publisher_->publish(DecodeYoloV8Values(
      message.header(), values.data(), values.size(), tensor.shape(), config_));
  } catch (const std::exception & error) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "YOLOv8 Managed HIP decoder dropped frame: %s", error.what());
  }
}

}  // namespace gpu_ros::yolov8

RCLCPP_COMPONENTS_REGISTER_NODE(
  gpu_ros::yolov8::YoloV8ManagedHipImageEncoderNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  gpu_ros::yolov8::YoloV8ManagedHipDecoderNode)
