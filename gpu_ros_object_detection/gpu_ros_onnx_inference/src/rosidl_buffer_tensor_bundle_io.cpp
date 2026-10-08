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

#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#ifdef GPU_ROS_ROSIDL_CUDA
#include "gpu_ros_rosidl_buffer/cuda_buffer_access.hpp"
#endif
namespace gpu_ros::onnx_inference
{
namespace
{
using Message = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
class BufferIO final : public ITensorBundleIO
{
public:
  BufferIO(rclcpp::Node * node, bool publish) : node_(node),
    provider_(ParseExecutionProvider(node->get_parameter("execution_provider").as_string())),
    device_(static_cast<int>(node->get_parameter("gpu_device_id").as_int())),
    allocator_(CreateBufferOutputAllocator(provider_, device_))
  {
    cpu_access_ = gpu_ros::rosidl_buffer::CreateCpuBufferAccess();
#ifdef GPU_ROS_ROSIDL_CUDA
    if (provider_ == ExecutionProvider::kCuda) {
      device_access_ = gpu_ros::rosidl_buffer::CreateCudaBufferAccess(device_);
    }
#endif
    if (publish) { publisher_ = node_->create_publisher<Message>("tensor_output", 10); }
  }
  void Subscribe(Callback callback) override
  {
    subscriber_ = node_->create_subscription<Message>("tensor_input", 10,
      [provider = provider_, device = device_, callback = std::move(callback),
        cpu = cpu_access_, access = device_access_, logger = node_->get_logger()](Message::ConstSharedPtr message) {
        try { callback(BindTensorBundle(std::move(message), provider, device, *cpu, access.get())); }
        catch (const std::exception & error) { RCLCPP_ERROR(logger, "Dropping Buffer input: %s", error.what()); }
      });
  }
  OutputPlacement output_placement() const noexcept override
  { return provider_ == ExecutionProvider::kCpu ? OutputPlacement::kHost : OutputPlacement::kDevice; }
  DeviceOutputAllocator * device_output_allocator() noexcept override { return allocator_.get(); }
  void Publish(TensorBundleOutput && output) override
  {
    if (!publisher_) { throw std::logic_error("Buffer publisher is disabled"); }
    auto message = BufferOutputMessage(output.tensors);
    // Move the original Buffers into the unique ROS envelope. Never copy/clone
    // a Buffer through rclcpp's intra-process const-reference overload.
    auto unique = std::make_unique<Message>(std::move(*message));
    unique->header = std::move(output.header);
    publisher_->publish(std::move(unique));
  }
private:
  rclcpp::Node * node_;
  ExecutionProvider provider_;
  int device_;
  std::unique_ptr<DeviceOutputAllocator> allocator_;
  std::shared_ptr<gpu_ros::rosidl_buffer::IBufferAccess> cpu_access_;
  std::shared_ptr<gpu_ros::rosidl_buffer::IBufferAccess> device_access_;
  rclcpp::Publisher<Message>::SharedPtr publisher_;
  rclcpp::Subscription<Message>::SharedPtr subscriber_;
};
}
std::unique_ptr<ITensorBundleIO> CreateRosidlBufferTensorBundleIO(rclcpp::Node * node, bool publish)
{ return std::make_unique<BufferIO>(node, publish); }
} // namespace gpu_ros::onnx_inference
