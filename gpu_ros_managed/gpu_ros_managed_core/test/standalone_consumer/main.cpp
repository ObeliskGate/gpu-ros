#include "gpu_ros_managed_core/buffer.hpp"

int main()
{
  return gpu_ros_managed::detail::pending_release_count(gpu_ros_managed::BackendKind::kCuda) == 0
           ? 0
           : 1;
}
