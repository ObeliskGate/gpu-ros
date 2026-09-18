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

#include "gpu_ros_detection_common/hip_preprocess.hpp"

#include <cmath>
#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>

namespace gpu_ros::detection_common
{
namespace
{

void CheckHip(hipError_t error, const char * operation)
{
  if (error == hipSuccess) {
    return;
  }
  std::ostringstream message;
  message << operation << ": " << hipGetErrorName(error) << " (" << hipGetErrorString(error) << ")";
  throw std::runtime_error(message.str());
}

__device__ float ReadRgb(
  const uint8_t * raw, int x, int y, int channel, const ImagePreprocessPlan plan)
{
  const uint8_t * row = raw + static_cast<size_t>(y) * plan.input_step;
  if (plan.encoding == ImageEncoding::kMono8) {
    return static_cast<float>(row[x]);
  }
  const int channels =
    plan.encoding == ImageEncoding::kRgb8 || plan.encoding == ImageEncoding::kBgr8 ? 3 : 4;
  const uint8_t * source = row + static_cast<size_t>(x) * channels;
  if (plan.encoding == ImageEncoding::kRgb8 || plan.encoding == ImageEncoding::kRgba8) {
    return static_cast<float>(source[channel]);
  }
  return static_cast<float>(source[2 - channel]);
}

__device__ void ResizeCoordinate(
  int destination, int source_size, int destination_size, int * source_index, float * weight)
{
  constexpr float kInterTabSize = 32.0F; // OpenCV INTER_BITS.
  const float source_coordinate = (static_cast<float>(destination) + 0.5F) *
                                    static_cast<float>(source_size) /
                                    static_cast<float>(destination_size) -
                                  0.5F;
  int index = static_cast<int>(floorf(source_coordinate));
  float fraction = source_coordinate - static_cast<float>(index);
  // OpenCV clamps a sample outside the source image to the edge and clears
  // the corresponding interpolation weight before applying its 5-bit
  // interpolation table.
  if (index < 0) {
    index = 0;
    fraction = 0.0F;
  } else if (index >= source_size - 1) {
    index = source_size - 1;
    fraction = 0.0F;
  } else {
    fraction = floorf(fraction * kInterTabSize + 0.5F) / kInterTabSize;
  }
  *source_index = index;
  *weight = fraction;
}

__global__ void PreprocessKernel(const uint8_t * raw, float * output, ImagePreprocessPlan plan)
{
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int channel = static_cast<int>(blockIdx.z);
  if (x >= plan.output_width || y >= plan.output_height || channel >= 3) {
    return;
  }
  const size_t plane = static_cast<size_t>(plan.output_width) * plan.output_height;
  const size_t index =
    static_cast<size_t>(channel) * plane + static_cast<size_t>(y) * plan.output_width + x;
  if (x >= plan.resized_width || y >= plan.resized_height) {
    output[index] = 0.0F;
    return;
  }

  int x0 = 0;
  int y0 = 0;
  float wx = 0.0F;
  float wy = 0.0F;
  ResizeCoordinate(x, static_cast<int>(plan.input_width), plan.resized_width, &x0, &wx);
  ResizeCoordinate(y, static_cast<int>(plan.input_height), plan.resized_height, &y0, &wy);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  if (x1 >= static_cast<int>(plan.input_width)) {
    x1 = static_cast<int>(plan.input_width) - 1;
  }
  if (y1 >= static_cast<int>(plan.input_height)) {
    y1 = static_cast<int>(plan.input_height) - 1;
  }
  const float top =
    (1.0F - wx) * ReadRgb(raw, x0, y0, channel, plan) + wx * ReadRgb(raw, x1, y0, channel, plan);
  const float bottom =
    (1.0F - wx) * ReadRgb(raw, x0, y1, channel, plan) + wx * ReadRgb(raw, x1, y1, channel, plan);
  const float value = (1.0F - wy) * top + wy * bottom;
  // The CPU reference resizes an 8-bit image, so reproduce its final
  // saturating integer quantization before applying the model normalization.
  const float quantized = fminf(255.0F, fmaxf(0.0F, floorf(value + 0.5F)));
  output[index] =
    plan.normalization == PreprocessNormalization::kUnitRange ? quantized / 255.0F : quantized;
}

__global__ void WriteInt64PairKernel(int64_t * output, int64_t first, int64_t second)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    output[0] = first;
    output[1] = second;
  }
}

} // namespace

void LaunchHipPreprocess(
  const uint8_t * raw_image, float * output, const ImagePreprocessPlan & plan, hipStream_t stream)
{
  if (raw_image == nullptr || output == nullptr || stream == nullptr) {
    throw std::invalid_argument("HIP preprocessing received a null pointer or stream");
  }
  const dim3 block(16, 16, 1);
  const dim3 grid((static_cast<unsigned>(plan.output_width) + block.x - 1U) / block.x,
    (static_cast<unsigned>(plan.output_height) + block.y - 1U) / block.y, 3U);
  hipLaunchKernelGGL(PreprocessKernel, grid, block, 0, stream, raw_image, output, plan);
  CheckHip(hipGetLastError(), "HIP image preprocess kernel launch");
}

void LaunchHipWriteInt64Pair(int64_t * output, int64_t first, int64_t second, hipStream_t stream)
{
  if (output == nullptr || stream == nullptr) {
    throw std::invalid_argument("HIP int64 writer received a null pointer or stream");
  }
  hipLaunchKernelGGL(WriteInt64PairKernel, dim3(1), dim3(1), 0, stream, output, first, second);
  CheckHip(hipGetLastError(), "HIP int64 pair kernel launch");
}

} // namespace gpu_ros::detection_common
