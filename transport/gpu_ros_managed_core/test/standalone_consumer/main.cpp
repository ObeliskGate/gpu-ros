#include "gpu_ros_managed_core/buffer.hpp"

int main()
{
  const auto readiness = gpu_ros_managed::BufferReadiness::kNotReady;
  return readiness == gpu_ros_managed::BufferReadiness::kNotReady ? 0 : 1;
}
