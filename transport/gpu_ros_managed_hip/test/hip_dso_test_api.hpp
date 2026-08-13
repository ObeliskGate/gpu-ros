// Copyright 2026 gpu_ros_managed contributors
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

#ifndef GPU_ROS_MANAGED_HIP__TEST__HIP_DSO_TEST_API_HPP_
#define GPU_ROS_MANAGED_HIP__TEST__HIP_DSO_TEST_API_HPP_

#include <memory>

#include "gpu_ros_managed_core/buffer.hpp"

std::shared_ptr<gpu_ros_managed::DeviceBuffer> MakeHipBufferInProducerDso();
void ReadHipBufferInConsumerDso(
  const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer);

#endif  // GPU_ROS_MANAGED_HIP__TEST__HIP_DSO_TEST_API_HPP_
