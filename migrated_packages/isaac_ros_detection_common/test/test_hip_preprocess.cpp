// Copyright 2026 Maintainer

#include "isaac_ros_detection_common/hip_preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

namespace
{

void CheckHip(hipError_t error, const char * operation)
{
  if (error != hipSuccess) {
    throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(error));
  }
}

struct HipAllocation
{
  void * pointer{nullptr};
  ~HipAllocation() {if (pointer != nullptr) {static_cast<void>(hipFree(pointer));}}
};

struct HipStream
{
  hipStream_t stream{nullptr};
  ~HipStream() {if (stream != nullptr) {static_cast<void>(hipStreamDestroy(stream));}}
};

sensor_msgs::msg::Image MakeImage(
  nvidia::isaac_ros::detection_common::ImageEncoding encoding)
{
  using nvidia::isaac_ros::detection_common::ImageEncoding;
  const uint32_t channels = encoding == ImageEncoding::kMono8 ? 1U :
    (encoding == ImageEncoding::kRgb8 || encoding == ImageEncoding::kBgr8 ? 3U : 4U);
  sensor_msgs::msg::Image image;
  image.width = 7;
  image.height = 5;
  image.step = image.width * channels + 5U;
  image.encoding = nvidia::isaac_ros::detection_common::ImageEncodingName(encoding);
  image.data.resize(static_cast<size_t>(image.step) * image.height, 0xEEU);
  for (uint32_t y = 0; y < image.height; ++y) {
    for (uint32_t x = 0; x < image.width; ++x) {
      for (uint32_t channel = 0; channel < channels; ++channel) {
        image.data[static_cast<size_t>(y) * image.step + x * channels + channel] =
          static_cast<uint8_t>((17U * x + 31U * y + 53U * channel + 7U) % 256U);
      }
    }
  }
  return image;
}

void CompareHipAndCpu(
  nvidia::isaac_ros::detection_common::ImageEncoding encoding,
  nvidia::isaac_ros::detection_common::PreprocessNormalization normalization,
  int64_t output_width = 11,
  int64_t output_height = 9,
  bool require_exact = false)
{
  using namespace nvidia::isaac_ros::detection_common;
  const auto image = MakeImage(encoding);
  const auto plan = MakeImagePreprocessPlan(image, output_width, output_height, normalization);
  const auto cpu = ExecuteCpuPreprocess(image, plan);

  HipStream stream;
  CheckHip(hipSetDevice(0), "hipSetDevice");
  CheckHip(hipStreamCreateWithFlags(&stream.stream, hipStreamNonBlocking), "hipStreamCreate");
  HipAllocation raw;
  HipAllocation output;
  CheckHip(hipMalloc(&raw.pointer, image.data.size()), "hipMalloc raw");
  CheckHip(hipMalloc(&output.pointer, cpu.size() * sizeof(float)), "hipMalloc output");
  CheckHip(hipMemcpyAsync(
      raw.pointer, image.data.data(), image.data.size(), hipMemcpyHostToDevice, stream.stream),
    "hipMemcpy raw");
  ASSERT_NO_THROW(LaunchHipPreprocess(
      static_cast<const uint8_t *>(raw.pointer), static_cast<float *>(output.pointer), plan,
      stream.stream));
  CheckHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize");
  std::vector<float> hip(cpu.size());
  CheckHip(hipMemcpy(
      hip.data(), output.pointer, hip.size() * sizeof(float), hipMemcpyDeviceToHost),
    "hipMemcpy output");

  const float tolerance = normalization == PreprocessNormalization::kUnitRange ?
    (1.0F / 255.0F + 1.0e-6F) : 1.0F;
  ASSERT_EQ(hip.size(), cpu.size());
  for (size_t index = 0; index < cpu.size(); ++index) {
    ASSERT_TRUE(std::isfinite(hip[index]));
    ASSERT_TRUE(std::isfinite(cpu[index]));
    if (require_exact) {
      EXPECT_FLOAT_EQ(hip[index], cpu[index]) << "index=" << index;
    } else {
      EXPECT_LE(std::abs(hip[index] - cpu[index]), tolerance) << "index=" << index;
    }
  }

  const size_t plane = static_cast<size_t>(plan.output_width) * plan.output_height;
  for (int channel = 0; channel < 3; ++channel) {
    for (int y = 0; y < plan.output_height; ++y) {
      for (int x = 0; x < plan.output_width; ++x) {
        if (x >= plan.resized_width || y >= plan.resized_height) {
          const size_t index = static_cast<size_t>(channel) * plane +
            static_cast<size_t>(y) * plan.output_width + x;
          EXPECT_FLOAT_EQ(hip[index], 0.0F);
        }
      }
    }
  }
}

}  // namespace

TEST(HipPreprocess, MatchesCpuReferenceAcrossSupportedEncodings)
{
  using namespace nvidia::isaac_ros::detection_common;
  for (const auto encoding : {
      ImageEncoding::kRgb8, ImageEncoding::kBgr8, ImageEncoding::kRgba8,
      ImageEncoding::kBgra8, ImageEncoding::kMono8})
  {
    SCOPED_TRACE(ImageEncodingName(encoding));
    CompareHipAndCpu(encoding, PreprocessNormalization::kNone);
    CompareHipAndCpu(encoding, PreprocessNormalization::kUnitRange);
  }
}

TEST(HipPreprocess, NoResizeChannelConversionIsElementwiseExact)
{
  using namespace nvidia::isaac_ros::detection_common;
  const auto image = MakeImage(ImageEncoding::kBgra8);
  EXPECT_NO_THROW(CompareHipAndCpu(
      ImageEncoding::kBgra8, PreprocessNormalization::kNone,
      image.width, image.height, true));
}
