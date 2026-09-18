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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <variant>
#include <vector>

#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#include "isaac_ros_nitros/types/cuda_stream_pool.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "gpu_ros_onnx_inference/nitros_managed_tensor_bundle_adapter.hpp"

namespace
{
namespace nitros = nvidia::isaac_ros::nitros;
namespace inference = gpu_ros::onnx_inference;

TEST(NitrosManagedTensorBundleAdapter, RoundTripPreservesDevicePayloadPointer)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto & nitros_stream_pool = nitros::CudaStreamPool::instance();
  const size_t available_streams_before = nitros_stream_pool.available();
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  void * device_memory = nullptr;
  ASSERT_EQ(cudaMalloc(&device_memory, 4U * sizeof(float)), cudaSuccess);

  {
    nitros::NitrosTensorListBuilder input_builder;
    input_builder.WithHeader(std_msgs::msg::Header{});
    input_builder.AddTensor(
      "images", nitros::NitrosTensorBuilder()
                  .WithShape(nitros::NitrosTensorShape({1, 1, 1, 4}))
                  .WithDataType(nitros::NitrosDataType::kFloat32)
                  .WithData(device_memory)
                  // The test owns the allocation. Production builders receive the managed
                  // buffer owner instead.
                  .WithReleaseCallback([] {})
                  .Build());
    auto nitros_input = input_builder.Build();
    const nitros::NitrosTensorListView input_view(nitros_input);

    auto input_read = nitros_input.get_read_handle(stream);
    const auto * input_pointer = input_read.get_ptr();
    ASSERT_NE(input_pointer, nullptr);

    inference::NitrosToManagedTensorBundleAdapter adapter(0);
    auto managed =
      std::make_shared<gpu_ros_managed::ManagedTensorBundle>(adapter.Convert(input_view));
    const auto & storage = managed->tensors().at(0).storage();
    const auto * buffer = std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&storage);
    ASSERT_NE(buffer, nullptr);
    auto managed_ready = (*buffer)->get_blocking_ready_lease();
    EXPECT_EQ(managed_ready.data(), input_pointer);

    auto nitros_output =
      inference::BuildNitrosTensorBundle(gpu_ros_managed::ManagedTensorBundleView(managed), 0);
    auto output_read = nitros_output.get_read_handle(stream);
    EXPECT_EQ(output_read.get_ptr(), input_pointer);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }

  const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (nitros_stream_pool.available() != available_streams_before &&
         std::chrono::steady_clock::now() < cleanup_deadline)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(nitros_stream_pool.available(), available_streams_before);
  EXPECT_TRUE(gpu_ros_managed::cuda::wait_for_pending_releases(std::chrono::seconds(5)));
  EXPECT_EQ(cudaFree(device_memory), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
