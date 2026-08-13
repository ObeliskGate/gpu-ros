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

#include "hip_dso_test_api.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>

#include "gpu_ros_managed_hip/hip_backend.hpp"

std::shared_ptr<gpu_ros_managed::DeviceBuffer> MakeHipBufferInProducerDso()
{
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return {};
  }
  const uint8_t value = 42U;
  auto buffer = gpu_ros_managed::hip::allocate(sizeof(value), 0);
  buffer->copy_from_host_blocking(&value, sizeof(value));
  return buffer;
}
