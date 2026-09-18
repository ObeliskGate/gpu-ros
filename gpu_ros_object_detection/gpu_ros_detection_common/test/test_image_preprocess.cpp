// Copyright 2026 Boshen Chen
#include "gpu_ros_detection_common/image_preprocess.hpp"

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
