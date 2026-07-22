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

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "isaac_ros_onnx_inference/tensor_list_io.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace nvidia::isaac_ros::onnx_inference
{

namespace
{

namespace nitros = nvidia::isaac_ros::nitros;
using nitros::PrimitiveType;

const std::string & Format()
{
  static const std::string format =
    nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name;
  return format;
}

ONNXTensorElementDataType GxfPrimitiveToOnnx(PrimitiveType pt)
{
  switch (pt) {
    case PrimitiveType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case PrimitiveType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(
              "NitrosTensorListIO: unsupported input PrimitiveType " +
              std::to_string(static_cast<int>(pt)));
  }
}

nitros::NitrosDataType OnnxToNitrosDtype(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return nitros::NitrosDataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return nitros::NitrosDataType::kInt64;
    default:
      throw std::runtime_error(
              "NitrosTensorListIO: unsupported output dtype " +
              std::to_string(static_cast<int>(dtype)));
  }
}

std::vector<int32_t> ToNitrosDims(const std::vector<int64_t> & shape)
{
  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (const int64_t dimension : shape) {
    if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("NitrosTensorListIO: tensor dimension is outside int32 range");
    }
    dims.push_back(static_cast<int32_t>(dimension));
  }
  return dims;
}

class NitrosTensorListIO : public ITensorListIO
{
public:
  explicit NitrosTensorListIO(rclcpp::Node * node)
  : node_{node},
    gpu_device_id_{node_->has_parameter("gpu_device_id") ?
      static_cast<int>(node_->get_parameter("gpu_device_id").as_int()) : 0}
  {
    pub_ = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>>(
      node_, "tensor_output", Format());
  }

  void Subscribe(Callback callback) override
  {
    callback_ = std::move(callback);
    sub_ = std::make_shared<nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>>(
      node_, "tensor_input", Format(),
      std::bind(&NitrosTensorListIO::OnView, this, std::placeholders::_1));
  }

  TensorMemoryKind OutputMemoryKind() const override
  {
    return TensorMemoryKind::kCudaDevice;
  }

  void Publish(
    std::vector<OwnedTensor> tensors,
    const std_msgs::msg::Header & header) override
  {
    nitros::NitrosTensorListBuilder builder;
    builder.WithHeader(header);
    for (auto & tensor : tensors) {
      auto * device_buffer = std::get_if<DeviceTensorBuffer>(&tensor.storage);
      if (device_buffer == nullptr || !device_buffer->owner) {
        throw std::runtime_error("NitrosTensorListIO requires an ORT-owned CUDA output tensor");
      }
      if (device_buffer->device_id != gpu_device_id_) {
        throw std::runtime_error(
                "NitrosTensorListIO output CUDA device does not match gpu_device_id");
      }
      auto owner = std::move(device_buffer->owner);
      builder.AddTensor(
        tensor.name,
        nitros::NitrosTensorBuilder()
        .WithShape(nitros::NitrosTensorShape(ToNitrosDims(tensor.shape)))
        .WithDataType(OnnxToNitrosDtype(tensor.dtype))
        .WithData(device_buffer->data)
        .WithReleaseCallback([owner = std::move(owner)]() mutable {owner.reset();})
        .Build());
    }
    pub_->publish(builder.Build());
  }

private:
  void OnView(const nitros::NitrosTensorListView & view)
  {
    std::vector<TensorView> inputs;
    inputs.reserve(view.GetTensorCount());
    for (const auto & t : view.GetAllTensor()) {
      TensorView tensor;
      tensor.name = t.GetName();
      tensor.dtype = GxfPrimitiveToOnnx(t.GetElementType());
      for (uint32_t i = 0; i < t.GetRank(); ++i) {
        tensor.shape.push_back(static_cast<int64_t>(t.GetDimension(i)));
      }
      tensor.data = t.GetBuffer();
      tensor.byte_size = t.GetTensorSize();
      tensor.memory_kind = TensorMemoryKind::kCudaDevice;
      tensor.device_id = gpu_device_id_;
      inputs.push_back(std::move(tensor));
    }

    std_msgs::msg::Header header;
    header.stamp.sec = view.GetTimestampSeconds();
    header.stamp.nanosec = view.GetTimestampNanoseconds();
    header.frame_id = view.GetFrameId();
    callback_(inputs, header);
  }

  rclcpp::Node * node_;
  Callback callback_;
  int gpu_device_id_;
  std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosTensorList>> pub_;
  std::shared_ptr<nitros::ManagedNitrosSubscriber<nitros::NitrosTensorListView>> sub_;
};

}  // namespace

std::unique_ptr<ITensorListIO> CreateNitrosTensorListIO(rclcpp::Node * node)
{
  return std::make_unique<NitrosTensorListIO>(node);
}

}  // namespace nvidia::isaac_ros::onnx_inference
