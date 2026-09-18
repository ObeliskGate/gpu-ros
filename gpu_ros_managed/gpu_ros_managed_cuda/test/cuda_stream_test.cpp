#include <cuda_runtime_api.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

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
  const std::vector<uint8_t> host_source{1, 2, 3, 4};
  buffer->copy_from_host_blocking(host_source.data(), host_source.size());
  std::vector<uint8_t> host_destination(host_source.size());
  buffer->copy_to_host_blocking(host_destination.data(), host_destination.size());
  assert(host_destination == host_source);
  buffer = gpu_ros_managed::cuda::allocate(4096, 0);
  {
    auto writer = buffer->get_write_handle(producer);
    assert(cudaMemsetAsync(writer.data(), 0x2a, writer.size(), producer.get()) == cudaSuccess);
    writer.finalize();
  }
  uint8_t first_a = 0;
  uint8_t first_b = 0;
  {
    auto reader = buffer->get_read_handle(consumer_a);
    assert(cudaMemcpyAsync(&first_a, reader.data(), 1, cudaMemcpyDeviceToHost, consumer_a.get()) ==
           cudaSuccess);
  }
  {
    auto reader = buffer->get_read_handle(consumer_b);
    assert(cudaMemcpyAsync(&first_b, reader.data(), 1, cudaMemcpyDeviceToHost, consumer_b.get()) ==
           cudaSuccess);
  }
  assert(cudaStreamSynchronize(consumer_a.get()) == cudaSuccess);
  assert(cudaStreamSynchronize(consumer_b.get()) == cudaSuccess);
  assert(first_a == 0x2a);
  assert(first_b == 0x2a);
  buffer.reset();
  assert(gpu_ros_managed::cuda::wait_for_pending_releases(std::chrono::seconds(2)));

  int external_owner_releases = 0;
  void * external_memory = nullptr;
  assert(cudaMalloc(&external_memory, 64) == cudaSuccess);
  auto external_owner =
    std::shared_ptr<void>(external_memory, [&external_owner_releases](void * p) {
      ++external_owner_releases;
      static_cast<void>(cudaFree(p));
    });
  auto externally_owned =
    gpu_ros_managed::cuda::adopt_synchronized_external(external_memory, 64, 0, external_owner);
  external_owner.reset();
  assert(external_owner_releases == 0);
  externally_owned.reset();
  assert(gpu_ros_managed::cuda::wait_for_pending_releases(std::chrono::seconds(2)));
  assert(external_owner_releases == 1);

  if (count > 1) {
    auto wrong_device_stream = gpu_ros_managed::cuda::make_stream(1);
    auto device_zero_buffer = gpu_ros_managed::cuda::allocate(64, 0);
    {
      auto writer = device_zero_buffer->get_write_handle(producer);
      writer.finalize();
    }
    bool rejected = false;
    try {
      static_cast<void>(device_zero_buffer->get_read_handle(wrong_device_stream));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
  }
  return 0;
}
