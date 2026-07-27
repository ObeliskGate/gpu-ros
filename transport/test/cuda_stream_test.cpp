#include <cuda_runtime_api.h>

#include <cassert>
#include <chrono>
#include <cstdint>

#include "gpu_ros_managed_cuda/cuda_backend.hpp"

int main()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return 77;
  }
  auto producer = gpu_ros_managed::cuda::make_stream(0);
  auto consumer_a = gpu_ros_managed::cuda::make_stream(0);
  auto consumer_b = gpu_ros_managed::cuda::make_stream(0);
  auto buffer = gpu_ros_managed::cuda::allocate(4096, 0);
  {
    auto writer = buffer->get_write_handle(producer);
    assert(cudaMemsetAsync(writer.data(), 0x2a, writer.size(), producer.get()) == cudaSuccess);
    writer.finalize();
  }
  uint8_t first_a = 0;
  uint8_t first_b = 0;
  {
    auto reader = buffer->get_read_handle(consumer_a);
    assert(cudaMemcpyAsync(
        &first_a, reader.data(), 1, cudaMemcpyDeviceToHost, consumer_a.get()) == cudaSuccess);
  }
  {
    auto reader = buffer->get_read_handle(consumer_b);
    assert(cudaMemcpyAsync(
        &first_b, reader.data(), 1, cudaMemcpyDeviceToHost, consumer_b.get()) == cudaSuccess);
  }
  assert(cudaStreamSynchronize(consumer_a.get()) == cudaSuccess);
  assert(cudaStreamSynchronize(consumer_b.get()) == cudaSuccess);
  assert(first_a == 0x2a);
  assert(first_b == 0x2a);
  buffer.reset();
  assert(gpu_ros_managed::cuda::wait_for_pending_releases(std::chrono::seconds(2)));
  return 0;
}
