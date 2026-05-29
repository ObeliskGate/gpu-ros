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

// Transport bridge: forwards a TensorList from a std-ROS2 (or NITROS) input to
// a NITROS output, without inference. Needed to feed NITROS-only vendor nodes
// (e.g. TensorRTNode) from a std-ROS2 publisher, since std pub -> NITROS sub is
// not bridged automatically (only the reverse direction is).

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{
namespace nitros = nvidia::isaac_ros::nitros;

nitros::NitrosDataType OnnxToNitrosDtype(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return nitros::NitrosDataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return nitros::NitrosDataType::kInt64;
    default:
      throw std::runtime_error("TensorListBridge: unsupported dtype");
  }
}
}  // namespace

// Subscribes via the configured input transport and republishes over NITROS.
class TensorListBridgeNode : public rclcpp::Node
{
public:
  explicit TensorListBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("tensor_list_bridge_node", options)
  {
    const std::string input_transport =
      declare_parameter<std::string>("input_transport", "std");

    cudaStreamCreate(&stream_);
    pub_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      this, "tensor_output",
      nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name);

    input_io_ = CreateTensorListIO(this, input_transport);
    input_io_->Subscribe(
      [this](const std::vector<HostTensor> & tensors, const std_msgs::msg::Header & header) {
        Forward(tensors, header);
      });
  }

  ~TensorListBridgeNode() override
  {
    cudaStreamDestroy(stream_);
  }

private:
  void Forward(
    const std::vector<HostTensor> & tensors, const std_msgs::msg::Header & header)
  {
    nitros::NitrosTensorListBuilder builder;
    builder.WithHeader(header);
    for (const auto & ht : tensors) {
      void * gpu_buffer = nullptr;
      cudaMallocAsync(&gpu_buffer, ht.data.size(), stream_);
      cudaMemcpyAsync(
        gpu_buffer, ht.data.data(), ht.data.size(), cudaMemcpyHostToDevice, stream_);
      std::vector<int32_t> dims(ht.shape.begin(), ht.shape.end());
      builder.AddTensor(
        ht.name,
        nitros::NitrosTensorBuilder()
        .WithShape(nitros::NitrosTensorShape(dims))
        .WithDataType(OnnxToNitrosDtype(ht.dtype))
        .WithData(gpu_buffer)
        .Build());
    }
    cudaStreamSynchronize(stream_);
    pub_->publish(builder.Build());
  }

  cudaStream_t stream_;
  std::unique_ptr<ITensorListIO> input_io_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> pub_;
};

}  // namespace nvidia::isaac_ros::onnx_inference

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::onnx_inference::TensorListBridgeNode)
