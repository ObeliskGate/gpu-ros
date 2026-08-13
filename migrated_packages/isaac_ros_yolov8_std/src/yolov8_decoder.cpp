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

#include "isaac_ros_yolov8_std/yolov8_decoder.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace nvidia::isaac_ros::yolov8_std
{
namespace
{
using Tensor = isaac_ros_tensor_list_interfaces::msg::Tensor;
using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

const Tensor & FindTensor(const TensorList & message, const std::string & name)
{
  for (const auto & tensor : message.tensors) {
    if (tensor.name == name) {return tensor;}
  }
  throw std::runtime_error("YOLOv8 tensor '" + name + "' not found");
}

size_t ElementCount(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) >
      std::numeric_limits<size_t>::max())
    {
      throw std::runtime_error("YOLOv8 output tensor has an invalid dimension");
    }
    const auto value = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / value) {
      throw std::overflow_error("YOLOv8 output tensor element count overflows size_t");
    }
    count *= value;
  }
  return count;
}

}  // namespace

vision_msgs::msg::Detection2DArray DecodeYoloV8Values(
  const std_msgs::msg::Header & header,
  const float * values,
  size_t value_count,
  const std::vector<int64_t> & shape,
  const YoloV8DecoderConfig & config)
{
  if (config.num_classes <= 0) {
    throw std::runtime_error("YOLOv8 num_classes must be positive");
  }
  if (config.confidence_threshold < 0.0 || config.confidence_threshold > 1.0) {
    throw std::runtime_error("YOLOv8 confidence_threshold must be in [0, 1]");
  }
  if (config.nms_threshold < 0.0 || config.nms_threshold > 1.0) {
    throw std::runtime_error("YOLOv8 nms_threshold must be in [0, 1]");
  }
  if (shape.size() != 3U || shape[0] != 1 || values == nullptr) {
    throw std::runtime_error(
            "YOLOv8 tensor must have shape [1, 4 + classes, boxes]");
  }
  const int64_t channels = shape[1];
  const int64_t num_boxes = shape[2];
  const int64_t expected_channels = 4 + config.num_classes;
  if (channels != expected_channels || num_boxes <= 0) {
    throw std::runtime_error(
            "YOLOv8 tensor shape does not match 4 + num_classes and a positive box count");
  }
  const size_t expected_count = ElementCount(shape);
  if (value_count != expected_count) {
    throw std::runtime_error("YOLOv8 typed span size does not match its shape");
  }

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> classes;
  bboxes.reserve(static_cast<size_t>(num_boxes));
  scores.reserve(static_cast<size_t>(num_boxes));
  classes.reserve(static_cast<size_t>(num_boxes));
  const auto value_at = [values, num_boxes](int64_t channel, int64_t box) {
      return values[static_cast<size_t>(channel * num_boxes + box)];
    };

  for (int64_t box = 0; box < num_boxes; ++box) {
    const double x = value_at(0, box);
    const double y = value_at(1, box);
    const double width = value_at(2, box);
    const double height = value_at(3, box);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
      !std::isfinite(height) || width <= 0.0 || height <= 0.0)
    {
      continue;
    }
    int class_id = 0;
    float score = value_at(4, box);
    for (int64_t class_index = 1; class_index < config.num_classes; ++class_index) {
      const float class_score = value_at(4 + class_index, box);
      if (class_score > score) {
        score = class_score;
        class_id = static_cast<int>(class_index);
      }
    }
    if (!std::isfinite(score)) {
      continue;
    }
    bboxes.emplace_back(x - (0.5 * width), y - (0.5 * height), width, height);
    scores.push_back(score);
    classes.push_back(class_id);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(
    bboxes, scores, static_cast<float>(config.confidence_threshold),
    static_cast<float>(config.nms_threshold), indices, 5.0F);
  vision_msgs::msg::Detection2DArray detections;
  detections.header = header;
  detections.detections.reserve(indices.size());
  for (const int index : indices) {
    const auto & bbox = bboxes.at(static_cast<size_t>(index));
    vision_msgs::msg::Detection2D detection;
    detection.header = header;
    detection.bbox.center.position.x = bbox.x + (0.5 * bbox.width);
    detection.bbox.center.position.y = bbox.y + (0.5 * bbox.height);
    detection.bbox.size_x = bbox.width;
    detection.bbox.size_y = bbox.height;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::to_string(classes.at(static_cast<size_t>(index)));
    hypothesis.hypothesis.score = scores.at(static_cast<size_t>(index));
    detection.results.push_back(hypothesis);
    detections.detections.push_back(std::move(detection));
  }
  return detections;
}

vision_msgs::msg::Detection2DArray DecodeYoloV8TensorList(
  const TensorList & message,
  const YoloV8DecoderConfig & config)
{
  const auto & tensor = FindTensor(message, config.tensor_name);
  if (tensor.data_type != kGxfFloat32 || tensor.shape.rank != 3U ||
    tensor.shape.dims.size() != 3U || tensor.data.size() % sizeof(float) != 0U)
  {
    throw std::runtime_error("YOLOv8 TensorList output has the wrong dtype, rank, or byte size");
  }
  std::vector<int64_t> shape;
  shape.reserve(tensor.shape.dims.size());
  for (const auto dimension : tensor.shape.dims) {
    shape.push_back(dimension);
  }
  std::vector<float> values(tensor.data.size() / sizeof(float));
  std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
  return DecodeYoloV8Values(message.header, values.data(), values.size(), shape, config);
}

}  // namespace nvidia::isaac_ros::yolov8_std
