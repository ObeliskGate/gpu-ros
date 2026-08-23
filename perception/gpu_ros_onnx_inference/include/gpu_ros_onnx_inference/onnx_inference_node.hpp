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

#ifndef GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
#define GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "gpu_ros_onnx_inference/onnx_inference_core.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"

namespace gpu_ros::onnx_inference
{

class OnnxInferenceNode : public rclcpp::Node
{
public:
  explicit OnnxInferenceNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OnnxInferenceNode() override;

private:
  struct TimingRecord
  {
    int64_t message_id;
    int64_t callback_start_ns;
    int64_t lock_wait_ns;
    int64_t run_inference_ns;
    int64_t input_setup_ns;
    int64_t ort_session_run_ns;
    int64_t output_materialize_ns;
    int64_t publish_ns;
    int64_t total_ns;
    uint8_t status;
  };

  void OnTensors(gpu_ros_managed::ManagedTensorBundleView inputs);
  void FinalizeOrtProfile(const char * reason) noexcept;
  void TrackMessageId(int64_t message_id);
  void WriteTimingReport() noexcept;

  std::unique_ptr<OnnxInferenceCore> core_;
  std::unique_ptr<ITensorBundleIO> io_;
  // OnnxInferenceCore owns mutable binding/profiling state and is explicitly
  // not thread-safe. Keep the inference and publish sequence serialized even
  // when the node is hosted by component_container_mt.
  std::mutex inference_mutex_;
  size_t ort_profile_frames_{0};
  size_t inference_count_{0};
  bool output_probe_runtime_logged_{false};
  bool debug_message_flow_{false};
  bool has_last_message_id_{false};
  int64_t last_message_id_{0};
  std::string timing_report_path_;
  std::vector<TimingRecord> timing_records_;
};

}  // namespace gpu_ros::onnx_inference

#endif  // GPU_ROS_ONNX_INFERENCE__ONNX_INFERENCE_NODE_HPP_
