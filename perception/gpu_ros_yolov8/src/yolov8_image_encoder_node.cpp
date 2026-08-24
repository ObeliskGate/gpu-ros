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

#include "gpu_ros_yolov8/yolov8_image_encoder_node.hpp"

#include <chrono>
#include <cinttypes>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include "rclcpp_components/register_node_macro.hpp"

namespace gpu_ros::yolov8
{

namespace
{

using SteadyClock = std::chrono::steady_clock;

int64_t ToNanoseconds(SteadyClock::time_point timestamp)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    timestamp.time_since_epoch()).count();
}

int64_t DurationNanoseconds(
  SteadyClock::time_point start, SteadyClock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

}  // namespace

YoloV8ImageEncoderNode::YoloV8ImageEncoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_image_encoder_node", options)
{
  config_.tensor_name = declare_parameter<std::string>("tensor_name", "images");
  config_.output_width = declare_parameter<int64_t>("output_width", 640);
  config_.output_height = declare_parameter<int64_t>("output_height", 640);
  debug_message_flow_ = declare_parameter<bool>("debug_message_flow", false);
  timing_report_path_ = declare_parameter<std::string>("timing_report_path", "");
  if (!timing_report_path_.empty()) {
    timing_records_.reserve(16384);
  }
  if (
    config_.tensor_name.empty() || config_.output_width <= 0 || config_.output_height <= 0 ||
    config_.output_width > std::numeric_limits<int>::max() ||
    config_.output_height > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument(
            "YOLOv8 tensor_name must not be empty and output dimensions must be positive "
            "and fit in an int");
  }

  pub_ = create_publisher<TensorBundle>("encoded_tensor", 10);
  sub_ = create_subscription<Image>(
    "image", 10,
    std::bind(&YoloV8ImageEncoderNode::InputCallback, this, std::placeholders::_1));
}

YoloV8ImageEncoderNode::~YoloV8ImageEncoderNode()
{
  sub_.reset();
  WriteTimingReport();
}

void YoloV8ImageEncoderNode::InputCallback(const Image::ConstSharedPtr msg)
{
  const int64_t message_id = msg->header.stamp.sec;
  const auto callback_start = SteadyClock::now();
  auto encode_finished = callback_start;
  auto callback_finished = callback_start;
  uint8_t timing_status = 1;
  TrackMessageId(message_id);
  try {
    const auto encoded = EncodeYoloV8Image(*msg, config_);
    encode_finished = SteadyClock::now();
    timing_status = 2;
    pub_->publish(encoded);
    callback_finished = SteadyClock::now();
    timing_status = 0;
  } catch (const std::exception & error) {
    callback_finished = SteadyClock::now();
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "YOLOv8 image encoder dropped frame: %s", error.what());
  }

  if (!timing_report_path_.empty()) {
    const auto encode_end = timing_status == 1 ? callback_finished : encode_finished;
    const auto publish_start = encode_finished;
    const auto publish_end = timing_status == 1 ? encode_finished : callback_finished;
    timing_records_.push_back(TimingRecord{
        message_id,
        ToNanoseconds(callback_start),
        DurationNanoseconds(callback_start, encode_end),
        DurationNanoseconds(publish_start, publish_end),
        DurationNanoseconds(callback_start, callback_finished),
        timing_status});
  }
}

void YoloV8ImageEncoderNode::WriteTimingReport() noexcept
{
  if (timing_report_path_.empty()) {
    return;
  }

  std::ofstream output(timing_report_path_, std::ios::out | std::ios::trunc);
  if (!output) {
    RCLCPP_ERROR(
      get_logger(), "Failed to open image encoder timing report '%s'.",
      timing_report_path_.c_str());
    return;
  }

  output << "message_id,callback_start_ns,encode_ns,publish_ns,total_ns,status\n";
  for (const auto & record : timing_records_) {
    const char * status = record.status == 0 ? "ok" :
      (record.status == 1 ? "encode_error" : "publish_error");
    output << record.message_id << ',' << record.callback_start_ns << ',' << record.encode_ns <<
      ',' << record.publish_ns << ',' << record.total_ns << ',' << status << '\n';
  }
  if (!output) {
    RCLCPP_ERROR(
      get_logger(), "Failed while writing image encoder timing report '%s'.",
      timing_report_path_.c_str());
    return;
  }
  RCLCPP_INFO(
    get_logger(), "Image encoder timing report written to '%s' (%zu callbacks).",
    timing_report_path_.c_str(), timing_records_.size());
}

void YoloV8ImageEncoderNode::TrackMessageId(int64_t message_id)
{
  if (!debug_message_flow_) {
    return;
  }
  if (has_last_message_id_ && message_id > last_message_id_ + 1) {
    RCLCPP_WARN(
      get_logger(),
      "MESSAGE_FLOW_GAP stage=image_encoder_input previous=%" PRId64
      " current=%" PRId64 " missing=%" PRId64,
      last_message_id_, message_id, message_id - last_message_id_ - 1);
  } else if (has_last_message_id_ && message_id <= last_message_id_) {
    RCLCPP_INFO(
      get_logger(), "MESSAGE_FLOW_RESET stage=image_encoder_input previous=%" PRId64
      " current=%" PRId64,
      last_message_id_, message_id);
  }
  last_message_id_ = message_id;
  has_last_message_id_ = true;
}

}  // namespace gpu_ros::yolov8

RCLCPP_COMPONENTS_REGISTER_NODE(gpu_ros::yolov8::YoloV8ImageEncoderNode)
