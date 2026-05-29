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

#include "isaac_ros_rtdetr/rtdetr_preprocessor_node.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

namespace
{
constexpr int kGxfInt64 = 7;
}  // namespace

RtDetrPreprocessorNode::RtDetrPreprocessorNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_preprocessor_node", options),
  input_image_tensor_name_{declare_parameter<std::string>(
      "input_image_tensor_name", "input_tensor")},
  output_image_tensor_name_{declare_parameter<std::string>(
      "output_image_tensor_name", "images")},
  output_size_tensor_name_{declare_parameter<std::string>(
      "output_size_tensor_name", "orig_target_sizes")},
  image_height_{declare_parameter<int64_t>("image_height", 480)},
  image_width_{declare_parameter<int64_t>("image_width", 640)},
  use_max_dim_for_orig_size_{declare_parameter<bool>("use_max_dim_for_orig_size", true)}
{
  pub_ = create_publisher<TensorList>("tensor_pub", 10);
  sub_ = create_subscription<TensorList>(
    "encoded_tensor", 10,
    std::bind(&RtDetrPreprocessorNode::InputCallback, this, std::placeholders::_1));
}

void RtDetrPreprocessorNode::InputCallback(const TensorList::SharedPtr msg)
{
  // Locate the encoded image tensor by name.
  const isaac_ros_tensor_list_interfaces::msg::Tensor * image_tensor = nullptr;
  for (const auto & t : msg->tensors) {
    if (t.name == input_image_tensor_name_) {
      image_tensor = &t;
      break;
    }
  }
  if (image_tensor == nullptr) {
    RCLCPP_WARN(
      get_logger(), "Input tensor '%s' not found; dropping message.",
      input_image_tensor_name_.c_str());
    return;
  }

  const int64_t orig_width = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_width_;
  const int64_t orig_height = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_height_;
  const int64_t output_size[2]{orig_width, orig_height};

  TensorList out_msg;
  out_msg.header = msg->header;

  // Forward the image tensor unchanged, renamed to the model's input binding.
  isaac_ros_tensor_list_interfaces::msg::Tensor image_out = *image_tensor;
  image_out.name = output_image_tensor_name_;
  out_msg.tensors.push_back(image_out);

  // Append the orig_target_sizes int64 [1, 2] tensor.
  isaac_ros_tensor_list_interfaces::msg::Tensor size_out;
  size_out.name = output_size_tensor_name_;
  size_out.data_type = kGxfInt64;
  size_out.shape.rank = 2;
  size_out.shape.dims = {1, 2};
  size_out.data.resize(sizeof(output_size));
  std::memcpy(size_out.data.data(), output_size, sizeof(output_size));
  out_msg.tensors.push_back(size_out);

  pub_->publish(out_msg);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode)
