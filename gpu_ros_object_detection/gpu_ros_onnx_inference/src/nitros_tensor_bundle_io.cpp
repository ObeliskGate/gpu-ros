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
//
#include <condition_variable>
#include <mutex>
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{
namespace
{
class NativeTensorBundleIO final : public ITensorBundleIO
{
  struct CallbackState
  {
    std::mutex mutex;
    std::condition_variable cv;
    Callback callback;
    size_t active{0};
    bool stopping{false};
  };

public:
  NativeTensorBundleIO(rclcpp::Node * node, bool publish_output)
      : transport_(node,
          node->has_parameter("gpu_device_id")
            ? static_cast<int>(node->get_parameter("gpu_device_id").as_int())
            : node->declare_parameter<int>("gpu_device_id", 0),
          "tensor_output", publish_output),
        allocator_(transport_), state_(std::make_shared<CallbackState>())
  {
  }
  ~NativeTensorBundleIO() override
  {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->stopping = true;
    state_->cv.wait(lock, [this] { return state_->active == 0; });
  }
  void Subscribe(Callback callback) override
  {
    state_->callback = std::move(callback);
    transport_.Subscribe([state = state_](gpu_ros_managed::ManagedTensorBundleView input) {
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->stopping) {
          return;
        }
        ++state->active;
      }
      struct Guard
      {
        std::shared_ptr<CallbackState> state;
        ~Guard()
        {
          std::lock_guard<std::mutex> lock(state->mutex);
          --state->active;
          state->cv.notify_all();
        }
      } guard{state};
      state->callback(std::move(input));
    });
  }
  OutputPlacement output_placement() const noexcept override { return OutputPlacement::kDevice; }
  DeviceOutputAllocator * device_output_allocator() noexcept override { return &allocator_; }
  void Publish(TensorBundleOutput && output) override
  {
    transport_.Publish(ToManagedOutput(std::move(output)));
  }

private:
  gpu_ros::nvidia_tensor_bundle_compat::TensorListTransport transport_;
  NativeDeviceOutputAllocator allocator_;
  std::shared_ptr<CallbackState> state_;
};
} // namespace
std::unique_ptr<ITensorBundleIO> CreateNativeTensorBundleIO(
  rclcpp::Node * node, bool publish_output)
{
  return std::make_unique<NativeTensorBundleIO>(node, publish_output);
}
} // namespace gpu_ros::onnx_inference
