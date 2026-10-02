// Copyright 2026 Boshen Chen
#include "gpu_ros_detection_common/image_preprocess.hpp"

#include <cstddef>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(ImagePreprocessPlan, ComputesLetterboxGeometry)
{
  sensor_msgs::msg::Image image;
  image.width = 4;
  image.height = 2;
  image.step = 12;
  image.encoding = "rgb8";
  image.data.resize(image.step * image.height, 1U);
  const auto plan = gpu_ros::detection_common::MakeImagePreprocessPlan(
    image, 8, 8, gpu_ros::detection_common::PreprocessNormalization::kNone);
  EXPECT_EQ(plan.resized_width, 8);
  EXPECT_EQ(plan.resized_height, 4);
  EXPECT_EQ(gpu_ros::detection_common::SourceRowBytes(plan), 12U);
}

TEST(ImagePreprocessPlan, RejectsShortStepAndBuffer)
{
  sensor_msgs::msg::Image image;
  image.width = 4;
  image.height = 2;
  image.step = 11;
  image.encoding = "rgb8";
  image.data.resize(22);
  EXPECT_THROW(gpu_ros::detection_common::MakeImagePreprocessPlan(
                 image, 8, 8, gpu_ros::detection_common::PreprocessNormalization::kNone),
    std::invalid_argument);
}

TEST(ImagePreprocessPlan, CpuReferenceUsesRgbNchwAndExactZeroPadding)
{
  sensor_msgs::msg::Image image;
  image.width = 2;
  image.height = 1;
  image.step = 8;
  image.encoding = "bgr8";
  image.data = {10U, 20U, 30U, 40U, 50U, 60U, 99U, 99U};
  const auto plan = gpu_ros::detection_common::MakeImagePreprocessPlan(
    image, 2, 2, gpu_ros::detection_common::PreprocessNormalization::kUnitRange);
  const auto values = gpu_ros::detection_common::ExecuteCpuPreprocess(image, plan);

  ASSERT_EQ(values.size(), 12U);
  EXPECT_FLOAT_EQ(values[0], 30.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[1], 60.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[2], 0.0F);
  EXPECT_FLOAT_EQ(values[3], 0.0F);
  EXPECT_FLOAT_EQ(values[4], 20.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[5], 50.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[8], 10.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[9], 40.0F / 255.0F);
  EXPECT_FLOAT_EQ(values[10], 0.0F);
  EXPECT_FLOAT_EQ(values[11], 0.0F);
}

TEST(ImagePreprocessPlan, CpuReferenceResizesRgbRampAndPadsBottom)
{
  sensor_msgs::msg::Image image;
  image.width = 2;
  image.height = 1;
  image.step = 6;
  image.encoding = "rgb8";
  image.data = {0U, 40U, 255U, 255U, 160U, 0U};
  const auto plan = gpu_ros::detection_common::MakeImagePreprocessPlan(
    image, 11, 7, gpu_ros::detection_common::PreprocessNormalization::kUnitRange);

  ASSERT_EQ(plan.output_width, 11);
  ASSERT_EQ(plan.output_height, 7);
  ASSERT_EQ(plan.resized_width, 11);
  ASSERT_EQ(plan.resized_height, 6);
  const auto values = gpu_ros::detection_common::ExecuteCpuPreprocess(image, plan);
  constexpr std::size_t kPlaneSize = 11U * 7U;
  ASSERT_EQ(values.size(), 3U * kPlaneSize);

  // The resized singleton-height ramp retains its RGB endpoint values in CHW order.
  for (std::size_t y = 0; y < 6U; ++y) {
    const std::size_t row = y * 11U;
    EXPECT_FLOAT_EQ(values[row], 0.0F);
    EXPECT_FLOAT_EQ(values[row + 10U], 1.0F);
    EXPECT_FLOAT_EQ(values[kPlaneSize + row], 40.0F / 255.0F);
    EXPECT_FLOAT_EQ(values[kPlaneSize + row + 10U], 160.0F / 255.0F);
    EXPECT_FLOAT_EQ(values[2U * kPlaneSize + row], 1.0F);
    EXPECT_FLOAT_EQ(values[2U * kPlaneSize + row + 10U], 0.0F);
  }

  // The final canvas row is bottom padding and must remain exactly zero in every plane.
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    for (std::size_t x = 0; x < 11U; ++x) {
      EXPECT_EQ(values[channel * kPlaneSize + 6U * 11U + x], 0.0F);
    }
  }
}

TEST(ImagePreprocessPlan, CpuReferenceTransposedRampPadsRight)
{
  sensor_msgs::msg::Image image;
  image.width = 1;
  image.height = 2;
  image.step = 3;
  image.encoding = "rgb8";
  image.data = {12U, 80U, 240U, 200U, 40U, 8U};
  const auto plan = gpu_ros::detection_common::MakeImagePreprocessPlan(
    image, 7, 11, gpu_ros::detection_common::PreprocessNormalization::kUnitRange);

  ASSERT_EQ(plan.output_width, 7);
  ASSERT_EQ(plan.output_height, 11);
  ASSERT_EQ(plan.resized_width, 6);
  ASSERT_EQ(plan.resized_height, 11);
  const auto values = gpu_ros::detection_common::ExecuteCpuPreprocess(image, plan);
  constexpr std::size_t kPlaneSize = 7U * 11U;
  ASSERT_EQ(values.size(), 3U * kPlaneSize);

  // A singleton source width is replicated across the resized image; endpoints
  // at the top and bottom stay anchored to the corresponding RGB source pixels.
  for (std::size_t channel = 0; channel < 3U; ++channel) {
    const float top = static_cast<float>(image.data[channel]) / 255.0F;
    const float bottom = static_cast<float>(image.data[3U + channel]) / 255.0F;
    const std::size_t plane = channel * kPlaneSize;
    EXPECT_FLOAT_EQ(values[plane], top);
    EXPECT_FLOAT_EQ(values[plane + 10U * 7U + 5U], bottom);
    EXPECT_FLOAT_EQ(values[plane + 10U * 7U], bottom);
    for (std::size_t y = 0; y < 11U; ++y) {
      EXPECT_FLOAT_EQ(values[plane + y * 7U], values[plane + y * 7U + 5U]);
      EXPECT_EQ(values[plane + y * 7U + 6U], 0.0F);
    }
  }
}

TEST(ImagePreprocessPlan, CpuReferenceReplicatesMonoChannel)
{
  sensor_msgs::msg::Image image;
  image.width = 1;
  image.height = 1;
  image.step = 1;
  image.encoding = "mono8";
  image.data = {17U};
  const auto plan = gpu_ros::detection_common::MakeImagePreprocessPlan(
    image, 1, 1, gpu_ros::detection_common::PreprocessNormalization::kNone);
  const auto values = gpu_ros::detection_common::ExecuteCpuPreprocess(image, plan);
  ASSERT_EQ(values.size(), 3U);
  EXPECT_FLOAT_EQ(values[0], 17.0F);
  EXPECT_FLOAT_EQ(values[1], 17.0F);
  EXPECT_FLOAT_EQ(values[2], 17.0F);
}
