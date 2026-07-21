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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8_std
{

namespace
{

using Tensor = isaac_ros_tensor_list_interfaces::msg::Tensor;
using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

const Tensor & FindTensor(const TensorList & msg, const std::string & tensor_name)
{
  for (const auto & tensor : msg.tensors) {
    if (tensor.name == tensor_name) {
      return tensor;
    }
  }
  throw std::runtime_error("YOLOv8 tensor '" + tensor_name + "' not found");
}

size_t ElementCount(const std::vector<uint32_t> & dims)
{
  size_t count = 1;
  for (const auto dim : dims) {
    if (dim == 0) {
      throw std::runtime_error("YOLOv8 output tensor has non-positive dimension");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<float> TensorDataToFloatVector(const Tensor & tensor)
{
  if (tensor.data_type != kGxfFloat32) {
    throw std::runtime_error(
            "YOLOv8 tensor '" + tensor.name + "' has data_type " +
            std::to_string(tensor.data_type) + "; expected float32");
  }

  const auto & dims = tensor.shape.dims;
  if (tensor.shape.rank != 3 || dims.size() != 3) {
    throw std::runtime_error(
            "YOLOv8 tensor '" + tensor.name + "' must have shape [1, 4 + classes, boxes]");
  }

  const size_t expected_bytes = ElementCount(dims) * sizeof(float);
  if (tensor.data.size() != expected_bytes) {
    throw std::runtime_error(
            "YOLOv8 tensor '" + tensor.name + "' byte size " +
            std::to_string(tensor.data.size()) + " does not match shape byte size " +
            std::to_string(expected_bytes));
  }

  std::vector<float> output(tensor.data.size() / sizeof(float));
  std::memcpy(output.data(), tensor.data.data(), tensor.data.size());
  return output;
}

}  // namespace

vision_msgs::msg::Detection2DArray DecodeYoloV8TensorList(
  const TensorList & msg,
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

  const auto & tensor = FindTensor(msg, config.tensor_name);
  const auto results = TensorDataToFloatVector(tensor);

  const auto & dims = tensor.shape.dims;
  const int64_t batch_size = dims.at(0);
  const int64_t channels = dims.at(1);
  const int64_t num_boxes = dims.at(2);
  const int64_t expected_channels = 4 + config.num_classes;

  if (batch_size != 1) {
    throw std::runtime_error("YOLOv8 decoder only supports batch size 1");
  }
  if (channels != expected_channels) {
    throw std::runtime_error(
            "YOLOv8 tensor channel dimension " + std::to_string(channels) +
            " does not match 4 + num_classes (" + std::to_string(expected_channels) + ")");
  }

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> classes;
  bboxes.reserve(static_cast<size_t>(num_boxes));
  scores.reserve(static_cast<size_t>(num_boxes));
  classes.reserve(static_cast<size_t>(num_boxes));

  const auto value_at = [&results, num_boxes](int64_t channel, int64_t box_index) {
      return results.at(static_cast<size_t>(channel * num_boxes + box_index));
    };

  for (int64_t i = 0; i < num_boxes; ++i) {
    const double x = value_at(0, i);
    const double y = value_at(1, i);
    const double w = value_at(2, i);
    const double h = value_at(3, i);

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) ||
      !std::isfinite(h) || w <= 0.0 || h <= 0.0)
    {
      continue;
    }

    int class_id = 0;
    float score = value_at(4, i);
    for (int64_t class_index = 1; class_index < config.num_classes; ++class_index) {
      const float class_score = value_at(4 + class_index, i);
      if (class_score > score) {
        score = class_score;
        class_id = static_cast<int>(class_index);
      }
    }

    bboxes.emplace_back(x - (0.5 * w), y - (0.5 * h), w, h);
    scores.push_back(score);
    classes.push_back(class_id);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(
    bboxes, scores, static_cast<float>(config.confidence_threshold),
    static_cast<float>(config.nms_threshold), indices, 5.0f);

  vision_msgs::msg::Detection2DArray detections;
  detections.header = msg.header;
  detections.detections.reserve(indices.size());

  for (const auto index : indices) {
    const auto & bbox = bboxes.at(static_cast<size_t>(index));

    vision_msgs::msg::Detection2D detection;
    detection.header = msg.header;
    detection.bbox.center.position.x = bbox.x + (0.5 * bbox.width);
    detection.bbox.center.position.y = bbox.y + (0.5 * bbox.height);
    detection.bbox.size_x = bbox.width;
    detection.bbox.size_y = bbox.height;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = std::to_string(classes.at(static_cast<size_t>(index)));
    hypothesis.hypothesis.score = scores.at(static_cast<size_t>(index));
    detection.results.push_back(hypothesis);

    detections.detections.push_back(detection);
  }

  return detections;
}

}  // namespace yolov8_std
}  // namespace isaac_ros
}  // namespace nvidia
