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

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "gpu_ros_onnx_inference/tensor_contract.hpp"

namespace gpu_ros::onnx_inference
{

namespace
{

using TensorBundleMsg = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;

class StdTensorBundleIO : public ITensorBundleIO
{
public:
  explicit StdTensorBundleIO(rclcpp::Node * node, bool publish_output) : node_{node}
  {
    if (publish_output) {
      pub_ = node_->create_publisher<TensorBundleMsg>("tensor_output", 10);
    }
  }

  void Subscribe(Callback callback) override
  {
    sub_ = node_->create_subscription<TensorBundleMsg>(
      "tensor_input", 10, [callback = std::move(callback), logger = node_->get_logger()](TensorBundleMsg::ConstSharedPtr msg) {
        try { OnMsg(std::move(msg), callback); }
        catch (const std::exception & error) { RCLCPP_ERROR(logger, "Dropping standard tensor input: %s", error.what()); }
      });
  }

  OutputPlacement output_placement() const noexcept override { return OutputPlacement::kHost; }

  void Publish(TensorBundleOutput && output) override
  {
    if (!pub_) {
      throw std::logic_error("standard TensorBundle output is disabled for this IO");
    }
    auto loaned_message = pub_->borrow_loaned_message();
    auto & msg = loaned_message.get();
    msg.header = std::move(output.header);
    msg.tensors.reserve(output.tensors.size());
    for (auto & tensor : output.tensors) {
      auto * host_data = std::get_if<std::vector<uint8_t>>(&tensor.storage);
      if (host_data == nullptr) {
        throw std::runtime_error("StdTensorBundleIO received a device-memory output tensor");
      }
      gpu_ros_tensor_bundle_msgs::msg::Tensor t;
      t.name = std::move(tensor.name);
      t.data_type = OnnxToBundleDtype(tensor.dtype);
      t.shape.assign(tensor.shape.begin(), tensor.shape.end());
      t.data = std::move(*host_data);
      msg.tensors.push_back(std::move(t));
    }
    pub_->publish(std::move(loaned_message));
  }

private:
  static void OnMsg(TensorBundleMsg::ConstSharedPtr msg, const Callback & callback)
  {
    TensorBindingBatch inputs{msg->header, msg, {}};
    inputs.bindings.reserve(msg->tensors.size());
    for (const auto & t : msg->tensors) {
      auto shape = t.shape;
      std::shared_ptr<const void> owner = msg;
      const uint8_t * data;
      if (t.data.get_backend_type() == "cpu") {
        data = t.data.data();
      } else {
        auto materialized = std::make_shared<std::vector<uint8_t>>(t.data.to_vector());
        data = materialized->data();
        owner = std::move(materialized);
      }
      const auto dtype = BundleToOnnxDtype(t.data_type);
      const auto bytes = TensorByteSize(dtype, shape, t.name);
      if (bytes != t.data.size()) { throw std::invalid_argument("standard tensor byte mismatch"); }
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      inputs.bindings.push_back({t.name, std::move(owner), data, bytes, "host",
        "standard message or host materialization through inference",
        Ort::Value::CreateTensor(memory, const_cast<uint8_t *>(data), bytes,
          shape.data(), shape.size(), dtype)});
    }
    callback(std::move(inputs));
  }

  rclcpp::Node * node_;
  rclcpp::Subscription<TensorBundleMsg>::SharedPtr sub_;
  rclcpp::Publisher<TensorBundleMsg>::SharedPtr pub_;
};

} // namespace

std::unique_ptr<ITensorBundleIO> CreateStdTensorBundleIO(rclcpp::Node * node, bool publish_output)
{
  return std::make_unique<StdTensorBundleIO>(node, publish_output);
}

} // namespace gpu_ros::onnx_inference
