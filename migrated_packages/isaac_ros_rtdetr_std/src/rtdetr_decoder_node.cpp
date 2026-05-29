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

#include "isaac_ros_rtdetr_std/rtdetr_decoder_node.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr_std
{

namespace
{

// Copy a named tensor's raw bytes into a typed vector. Data is already on host.
template<typename T>
std::vector<T> TensorToVector(
  const isaac_ros_tensor_list_interfaces::msg::TensorList & msg,
  const std::string & tensor_name)
{
  for (const auto & t : msg.tensors) {
    if (t.name == tensor_name) {
      std::vector<T> out(t.data.size() / sizeof(T));
      std::memcpy(out.data(), t.data.data(), t.data.size());
      return out;
    }
  }
  return {};
}

}  // namespace

RtDetrDecoderNode::RtDetrDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_decoder_node", options),
  labels_tensor_name_{declare_parameter<std::string>("labels_tensor_name", "labels")},
  boxes_tensor_name_{declare_parameter<std::string>("boxes_tensor_name", "boxes")},
  scores_tensor_name_{declare_parameter<std::string>("scores_tensor_name", "scores")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.9)}
{
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", 10);
  sub_ = create_subscription<TensorList>(
    "tensor_sub", 10,
    std::bind(&RtDetrDecoderNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrDecoderNode::InputCallback(const TensorList::SharedPtr msg)
{
  auto labels = TensorToVector<int64_t>(*msg, labels_tensor_name_);
  auto boxes = TensorToVector<float>(*msg, boxes_tensor_name_);
  auto scores = TensorToVector<float>(*msg, scores_tensor_name_);

  vision_msgs::msg::Detection2DArray detections;
  detections.header = msg->header;

  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores.at(i) <= confidence_threshold_) {
      continue;
    }

    vision_msgs::msg::Detection2D detection;
    detection.header = msg->header;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(labels.at(i));
    hyp.hypothesis.score = scores.at(i);
    detection.results.push_back(hyp);

    // Boxes are stored as 4 contiguous (x1, y1, x2, y2) values.
    constexpr size_t kBoxSize = 4;
    const float x1 = boxes.at(kBoxSize * i);
    const float y1 = boxes.at(kBoxSize * i + 1);
    const float x2 = boxes.at(kBoxSize * i + 2);
    const float y2 = boxes.at(kBoxSize * i + 3);

    detection.bbox.center.position.x = (x1 + x2) / 2;
    detection.bbox.center.position.y = (y1 + y2) / 2;
    detection.bbox.size_x = (x2 - x1);
    detection.bbox.size_y = (y2 - y1);

    detections.detections.push_back(detection);
  }

  pub_->publish(detections);
}

}  // namespace rtdetr_std
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr_std::RtDetrDecoderNode)
