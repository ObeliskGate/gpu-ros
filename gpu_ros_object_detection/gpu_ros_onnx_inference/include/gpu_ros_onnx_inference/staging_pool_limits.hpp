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

#ifndef GPU_ROS_ONNX_INFERENCE__STAGING_POOL_LIMITS_HPP_
#define GPU_ROS_ONNX_INFERENCE__STAGING_POOL_LIMITS_HPP_

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_ros::onnx_inference::detail
{

inline size_t ValidateStagingPoolAddition(size_t tensor_bytes, size_t pool_capacity,
  size_t current_cache_bytes, size_t current_cache_entries, size_t max_tensor_bytes,
  size_t max_cache_bytes, size_t max_cache_entries)
{
  if (tensor_bytes == 0U || tensor_bytes > max_tensor_bytes) {
    throw std::runtime_error("tensor requires " + std::to_string(tensor_bytes) +
                             " bytes; adjust managed_max_tensor_bytes if this shape is expected");
  }
  if (current_cache_entries >= max_cache_entries) {
    throw std::runtime_error("dynamic staging pool cache reached managed_pool_cache_max_entries=" +
                             std::to_string(max_cache_entries));
  }
  if (tensor_bytes > std::numeric_limits<size_t>::max() / pool_capacity) {
    throw std::overflow_error("dynamic staging pool allocation overflows size_t");
  }
  const size_t pool_bytes = tensor_bytes * pool_capacity;
  if (pool_bytes > std::numeric_limits<size_t>::max() - current_cache_bytes) {
    throw std::overflow_error("dynamic staging pool cache overflows size_t");
  }
  const size_t next_cache_bytes = current_cache_bytes + pool_bytes;
  if (next_cache_bytes > max_cache_bytes) {
    throw std::runtime_error(
      "dynamic staging pools would require " + std::to_string(next_cache_bytes) +
      " bytes; adjust managed_pool_cache_max_bytes=" + std::to_string(max_cache_bytes));
  }
  return next_cache_bytes;
}

} // namespace gpu_ros::onnx_inference::detail

#endif // GPU_ROS_ONNX_INFERENCE__STAGING_POOL_LIMITS_HPP_
