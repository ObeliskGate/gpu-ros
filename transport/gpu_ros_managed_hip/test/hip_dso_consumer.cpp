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

#include "hip_dso_test_api.hpp"

#include <memory>

#include "gpu_ros_managed_hip/hip_backend.hpp"

void ReadHipBufferInConsumerDso(
  const std::shared_ptr<gpu_ros_managed::DeviceBuffer> & buffer)
{
  auto stream = gpu_ros_managed::hip::make_stream(0);
  auto read = buffer->get_read_handle(stream.stream());
  read.finish();
}
