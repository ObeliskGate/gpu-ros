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

#ifndef GPU_ROS_ONNX_INFERENCE__ONNX_SESSION_HPP_
#define GPU_ROS_ONNX_INFERENCE__ONNX_SESSION_HPP_

#include <memory>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"

namespace gpu_ros::onnx_inference
{
enum class ExecutionProvider { kCuda, kRocm, kMigraphx, kCpu };
ExecutionProvider ParseExecutionProvider(const std::string & ep_str);

// ORT allocation owners must retain this object until their values are destroyed.
class OnnxSession
{
public:
  struct Config
  {
    std::string model_file_path;
    ExecutionProvider ep{ExecutionProvider::kCuda};
    int gpu_device_id{0};
    std::string ort_profile_prefix;
  };
  explicit OnnxSession(const Config & cfg);
  Ort::Session & session() noexcept { return *session_; }
  const std::vector<std::string> & input_names() const noexcept { return input_names_; }
  const std::vector<std::string> & output_names() const noexcept { return output_names_; }
  bool IsProfilingEnabled() const noexcept { return profiling_enabled_; }
  std::string EndProfiling();

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool profiling_enabled_{false};
};
} // namespace gpu_ros::onnx_inference
#endif // GPU_ROS_ONNX_INFERENCE__ONNX_SESSION_HPP_
