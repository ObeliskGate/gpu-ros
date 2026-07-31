// Copyright 2026 Maintainer
// Licensed under the Apache License, Version 2.0.
#include <memory>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_onnx_inference/nitros_managed_tensor_list_adapter.hpp"

namespace
{
namespace nitros = nvidia::isaac_ros::nitros;
namespace inference = nvidia::isaac_ros::onnx_inference;

TEST(NitrosManagedTensorListAdapter, RoundTripPreservesDevicePayloadPointer)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available.";
  }

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  void * device_memory = nullptr;
  ASSERT_EQ(cudaMalloc(&device_memory, 4U * sizeof(float)), cudaSuccess);

  {
    nitros::NitrosTensorListBuilder input_builder;
    input_builder.WithHeader(std_msgs::msg::Header{});
    input_builder.AddTensor(
      "images",
      nitros::NitrosTensorBuilder()
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

    inference::NitrosToManagedTensorListAdapter adapter(0);
    auto managed = std::make_shared<gpu_ros_managed::ManagedTensorList>(
      adapter.Convert(input_view));
    const auto & storage = managed->tensors().at(0).storage();
    const auto * buffer = std::get_if<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(&storage);
    ASSERT_NE(buffer, nullptr);
    auto managed_ready = (*buffer)->get_blocking_ready_lease();
    EXPECT_EQ(managed_ready.data(), input_pointer);

    auto nitros_output = inference::BuildNitrosTensorList(
      gpu_ros_managed::ManagedTensorListView(managed), 0);
    auto output_read = nitros_output.get_read_handle(stream);
    EXPECT_EQ(output_read.get_ptr(), input_pointer);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }

  EXPECT_EQ(cudaFree(device_memory), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
