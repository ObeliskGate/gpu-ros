// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.

#include <cassert>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "gpu_ros_managed_hip/hip_backend.hpp"

int main()
{
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    return 77;
  }

  const std::vector<uint8_t> source{9, 8, 7, 6, 5};
  auto buffer = gpu_ros_managed::hip::allocate(source.size(), 0);
  buffer->copy_from_host_blocking(source.data(), source.size());
  std::vector<uint8_t> destination(source.size());
  buffer->copy_to_host_blocking(destination.data(), destination.size());
  assert(destination == source);
  return 0;
}
