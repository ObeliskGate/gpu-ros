// Copyright 2026 Boshen Chen

#include "gpu_ros_detection_common/hip_preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
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
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
  }
}

struct HipAllocation
{
  void * pointer{nullptr};
  ~HipAllocation()
  {
    if (pointer != nullptr) {
      static_cast<void>(hipFree(pointer));
    }
  }
};

struct HipStream
{
  hipStream_t stream{nullptr};
  ~HipStream()
  {
    if (stream != nullptr) {
      static_cast<void>(hipStreamDestroy(stream));
    }
  }
};

sensor_msgs::msg::Image MakeImage(gpu_ros::detection_common::ImageEncoding encoding)
{
  using gpu_ros::detection_common::ImageEncoding;
  const uint32_t channels =
    encoding == ImageEncoding::kMono8
      ? 1U
      : (encoding == ImageEncoding::kRgb8 || encoding == ImageEncoding::kBgr8 ? 3U : 4U);
  sensor_msgs::msg::Image image;
  image.width = 7;
  image.height = 5;
  image.step = image.width * channels + 5U;
  image.encoding = gpu_ros::detection_common::ImageEncodingName(encoding);
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

sensor_msgs::msg::Image MakeMonoImage(
  uint32_t width, uint32_t height, const std::vector<uint8_t> & pixels)
{
  if (pixels.size() != static_cast<size_t>(width) * height) {
    throw std::invalid_argument("mono image pixel count does not match its dimensions");
  }
  sensor_msgs::msg::Image image;
  image.width = width;
  image.height = height;
  image.step = width + 3U;
  image.encoding = gpu_ros::detection_common::ImageEncodingName(
    gpu_ros::detection_common::ImageEncoding::kMono8);
  image.data.assign(static_cast<size_t>(image.step) * height, 0xEEU);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      image.data[static_cast<size_t>(y) * image.step + x] =
        pixels[static_cast<size_t>(y) * width + x];
    }
  }
  return image;
}

sensor_msgs::msg::Image MakeCheckerboard(uint32_t width, uint32_t height)
{
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      pixels[static_cast<size_t>(y) * width + x] = (x + y) % 2U == 0U ? 0U : 255U;
    }
  }
  return MakeMonoImage(width, height, pixels);
}

std::vector<float> RunHipPreprocess(const sensor_msgs::msg::Image & image,
  const gpu_ros::detection_common::ImagePreprocessPlan & plan, size_t output_count)
{
  HipStream stream;
  CheckHip(hipSetDevice(0), "hipSetDevice");
  CheckHip(hipStreamCreateWithFlags(&stream.stream, hipStreamNonBlocking), "hipStreamCreate");
  HipAllocation raw;
  HipAllocation output;
  CheckHip(hipMalloc(&raw.pointer, image.data.size()), "hipMalloc raw");
  CheckHip(hipMalloc(&output.pointer, output_count * sizeof(float)), "hipMalloc output");
  CheckHip(hipMemcpyAsync(raw.pointer, image.data.data(), image.data.size(), hipMemcpyHostToDevice,
             stream.stream),
    "hipMemcpy raw");
  LaunchHipPreprocess(static_cast<const uint8_t *>(raw.pointer),
    static_cast<float *>(output.pointer), plan, stream.stream);
  CheckHip(hipStreamSynchronize(stream.stream), "hipStreamSynchronize");
  std::vector<float> hip(output_count);
  CheckHip(hipMemcpy(hip.data(), output.pointer, hip.size() * sizeof(float), hipMemcpyDeviceToHost),
    "hipMemcpy output");
  return hip;
}

std::vector<float> CompareHipAndCpu(
  const sensor_msgs::msg::Image & image,
  gpu_ros::detection_common::PreprocessNormalization normalization, int64_t output_width = 11,
  int64_t output_height = 9, bool require_exact = false)
{
  using namespace gpu_ros::detection_common;
  const auto plan = MakeImagePreprocessPlan(image, output_width, output_height, normalization);
  const auto cpu = ExecuteCpuPreprocess(image, plan);
  std::vector<float> hip;
  try {
    hip = RunHipPreprocess(image, plan, cpu.size());
  } catch (const std::exception & error) {
    ADD_FAILURE() << "HIP preprocessing failed: " << error.what();
    return hip;
  }

  const float tolerance =
    normalization == PreprocessNormalization::kUnitRange ? (1.0F / 255.0F + 1.0e-6F) : 1.0F;
  EXPECT_EQ(hip.size(), cpu.size());
  if (hip.size() != cpu.size()) {
    return hip;
  }
  for (size_t index = 0; index < cpu.size(); ++index) {
    EXPECT_TRUE(std::isfinite(hip[index])) << "index=" << index;
    EXPECT_TRUE(std::isfinite(cpu[index])) << "index=" << index;
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
          const size_t index =
            static_cast<size_t>(channel) * plane + static_cast<size_t>(y) * plan.output_width + x;
          EXPECT_FLOAT_EQ(hip[index], 0.0F);
        }
      }
    }
  }
  return hip;
}

std::vector<float> CompareHipAndCpu(gpu_ros::detection_common::ImageEncoding encoding,
  gpu_ros::detection_common::PreprocessNormalization normalization, int64_t output_width = 11,
  int64_t output_height = 9, bool require_exact = false)
{
  return CompareHipAndCpu(
    MakeImage(encoding), normalization, output_width, output_height, require_exact);
}

} // namespace

TEST(HipPreprocess, MatchesCpuReferenceAcrossSupportedEncodings)
{
  using namespace gpu_ros::detection_common;
  for (const auto encoding : {ImageEncoding::kRgb8, ImageEncoding::kBgr8, ImageEncoding::kRgba8,
         ImageEncoding::kBgra8, ImageEncoding::kMono8})
  {
    SCOPED_TRACE(ImageEncodingName(encoding));
    CompareHipAndCpu(encoding, PreprocessNormalization::kNone);
    CompareHipAndCpu(encoding, PreprocessNormalization::kUnitRange);
  }
}

TEST(HipPreprocess, NoResizeChannelConversionIsElementwiseExact)
{
  using namespace gpu_ros::detection_common;
  const auto image = MakeImage(ImageEncoding::kBgra8);
  EXPECT_NO_THROW(CompareHipAndCpu(
    ImageEncoding::kBgra8, PreprocessNormalization::kNone, image.width, image.height, true));
}

TEST(HipPreprocess, ResizeTwoByOneRampMatchesCpuEndpointsAndZeroPadding)
{
  using namespace gpu_ros::detection_common;
  const auto image = MakeMonoImage(2U, 1U, {0U, 255U});
  const auto normalization = PreprocessNormalization::kNone;
  const auto plan = MakeImagePreprocessPlan(image, 11, 7, normalization);
  EXPECT_EQ(plan.resized_width, 11);
  EXPECT_EQ(plan.resized_height, 6);
  const auto hip = CompareHipAndCpu(image, normalization, 11, 7);
  ASSERT_EQ(hip.size(), 3U * 11U * 7U);

  const size_t plane = 11U * 7U;
  for (size_t channel = 0; channel < 3U; ++channel) {
    EXPECT_FLOAT_EQ(hip[channel * plane], 0.0F);
    EXPECT_FLOAT_EQ(hip[channel * plane + 10U], 255.0F);
    for (size_t x = 0; x < 11U; ++x) {
      EXPECT_FLOAT_EQ(hip[channel * plane + 6U * 11U + x], 0.0F);
    }
  }
}

TEST(HipPreprocess, ResizeTransposedRampMatchesCpuEndpointsAndZeroPadding)
{
  using namespace gpu_ros::detection_common;
  const auto image = MakeMonoImage(1U, 2U, {0U, 255U});
  const auto normalization = PreprocessNormalization::kNone;
  const auto plan = MakeImagePreprocessPlan(image, 7, 11, normalization);
  EXPECT_EQ(plan.resized_width, 6);
  EXPECT_EQ(plan.resized_height, 11);
  const auto hip = CompareHipAndCpu(image, normalization, 7, 11);
  ASSERT_EQ(hip.size(), 3U * 7U * 11U);

  const size_t plane = 7U * 11U;
  for (size_t channel = 0; channel < 3U; ++channel) {
    EXPECT_FLOAT_EQ(hip[channel * plane], 0.0F);
    EXPECT_FLOAT_EQ(hip[channel * plane + 10U * 7U], 255.0F);
    for (size_t y = 0; y < 11U; ++y) {
      EXPECT_FLOAT_EQ(hip[channel * plane + y * 7U + 6U], 0.0F);
    }
  }
}

TEST(HipPreprocess, ResizeSingletonClampsSamplesAndPadsExactly)
{
  using namespace gpu_ros::detection_common;
  const auto image = MakeMonoImage(1U, 1U, {137U});
  const auto normalization = PreprocessNormalization::kNone;
  const auto plan = MakeImagePreprocessPlan(image, 7, 5, normalization);
  EXPECT_EQ(plan.resized_width, 5);
  EXPECT_EQ(plan.resized_height, 5);
  const auto hip = CompareHipAndCpu(image, normalization, 7, 5);
  ASSERT_EQ(hip.size(), 3U * 7U * 5U);

  const size_t plane = 7U * 5U;
  for (size_t channel = 0; channel < 3U; ++channel) {
    EXPECT_FLOAT_EQ(hip[channel * plane], 137.0F);
    EXPECT_FLOAT_EQ(hip[channel * plane + 4U * 7U + 4U], 137.0F);
    for (size_t y = 0; y < 5U; ++y) {
      for (size_t x = 5U; x < 7U; ++x) {
        EXPECT_FLOAT_EQ(hip[channel * plane + y * 7U + x], 0.0F);
      }
    }
  }
}

TEST(HipPreprocess, ResizeNonIntegerDownscaleMatchesCpuReference)
{
  using namespace gpu_ros::detection_common;
  std::vector<uint8_t> pixels(11U * 7U);
  for (uint32_t y = 0; y < 7U; ++y) {
    for (uint32_t x = 0; x < 11U; ++x) {
      pixels[static_cast<size_t>(y) * 11U + x] =
        static_cast<uint8_t>((37U * x + 61U * y + 19U * x * y) % 256U);
    }
  }
  const auto image = MakeMonoImage(11U, 7U, pixels);
  const auto normalization = PreprocessNormalization::kNone;
  const auto plan = MakeImagePreprocessPlan(image, 5, 4, normalization);
  EXPECT_EQ(plan.resized_width, 5);
  EXPECT_EQ(plan.resized_height, 3);
  CompareHipAndCpu(image, normalization, 5, 4);
}

TEST(HipPreprocess, ResizeCheckerboardMatchesCpuReference)
{
  using namespace gpu_ros::detection_common;
  const auto image = MakeCheckerboard(11U, 7U);
  const auto normalization = PreprocessNormalization::kNone;
  const auto plan = MakeImagePreprocessPlan(image, 16, 11, normalization);
  EXPECT_EQ(plan.resized_width, 16);
  EXPECT_EQ(plan.resized_height, 10);
  CompareHipAndCpu(image, normalization, 16, 11);
}

TEST(HipPreprocess, ResizeRoundingNeighborhoodMatchesCpuReference)
{
  using namespace gpu_ros::detection_common;
  const auto normalization = PreprocessNormalization::kNone;
  for (const auto & pixels :
    {std::vector<uint8_t>{0U, 10U}, std::vector<uint8_t>{0U, 11U},
      std::vector<uint8_t>{0U, 12U}, std::vector<uint8_t>{10U, 0U},
      std::vector<uint8_t>{11U, 0U}, std::vector<uint8_t>{12U, 0U}})
  {
    SCOPED_TRACE(static_cast<int>(pixels[0]) * 256 + pixels[1]);
    const auto image = MakeMonoImage(2U, 1U, pixels);
    const auto hip = CompareHipAndCpu(image, normalization, 11, 7);
    ASSERT_EQ(hip.size(), 3U * 11U * 7U);
  }
}
