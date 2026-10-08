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

#include "model_path.hpp"
#include <filesystem>
#include <stdexcept>
namespace gpu_ros::onnx_inference
{
std::string ResolveModelProfile(const std::string & requested_profile,
  const std::string & execution_provider, const std::string & assets_root)
{
  const std::string profile =
    requested_profile == "auto"
      ? (execution_provider == "cuda" ? "nvidia_synthetica" : "rtdetrv2_r50")
      : requested_profile;
  std::filesystem::path relative;
  if (profile == "nvidia_synthetica") {
    relative = "models/synthetica_detr_v1.0.0_onnx/sdetr_grasp.onnx";
  } else if (profile == "rtdetrv2_r50") {
    relative = "models/rtdetrv2_r50/rtdetrv2_r50.onnx";
  } else if (profile == "xanylabeling_rtdetrv2_r50") {
    relative = "models/xanylabeling_rtdetrv2_r50/rtdetrv2_r50vd_6x_coco.onnx";
  } else {
    throw std::invalid_argument("model_profile must be auto, nvidia_synthetica, rtdetrv2_r50, or "
                                "xanylabeling_rtdetrv2_r50");
  }
  const auto path = std::filesystem::path(assets_root) / relative;
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("selected model_profile=" + profile + " is missing at " +
                             path.string() + "; automatic model fallback is disabled");
  }
  return path.string();
}
} // namespace gpu_ros::onnx_inference
