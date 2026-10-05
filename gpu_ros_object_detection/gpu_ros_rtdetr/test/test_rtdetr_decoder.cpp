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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu_ros_rtdetr/rtdetr_decoder.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor_bundle.hpp"
#include "rosidl_buffer/buffer.hpp"

template <typename T> class TestNonCpuBufferImpl final : public rosidl::BufferImplBase<T>
{
public:
  explicit TestNonCpuBufferImpl(std::vector<T> values) : values_(std::move(values)) {}

  std::string get_backend_type() const override { return "test_non_cpu"; }
  size_t size() const override { return values_.size(); }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage() = values_;
    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    return std::make_unique<TestNonCpuBufferImpl<T>>(values_);
  }

private:
  std::vector<T> values_;
};

template <typename T> rosidl::Buffer<T> MakeTestNonCpuBuffer(std::vector<T> values)
{
  return rosidl::Buffer<T>(std::make_unique<TestNonCpuBufferImpl<T>>(std::move(values)));
}

template <typename T> std::vector<uint8_t> BytesOf(const std::vector<T> & values)
{
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
namespace
{
using gpu_ros::rtdetr::DecodeRtDetrValues;
using gpu_ros::rtdetr::RtDetrDecoderConfig;

RtDetrDecoderConfig Config()
{
  RtDetrDecoderConfig config;
  config.confidence_threshold = 0.5;
  return config;
}
} // namespace
TEST(RtDetrDecoderTest, MaterializesNonCpuMetadataAndPayloadAtHostBoundary)
{
  using gpu_ros_tensor_bundle_msgs::msg::Tensor;
  using gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
  std::vector<int64_t> labels(300U, 0);
  std::vector<float> boxes(1200U, 0.0F);
  std::vector<float> scores(300U, 0.0F);
  labels[0] = 7;
  boxes[0] = 10.0F;
  boxes[1] = 20.0F;
  boxes[2] = 50.0F;
  boxes[3] = 80.0F;
  scores[0] = 0.9F;

  Tensor label_tensor;
  label_tensor.name = "labels";
  label_tensor.data_type = Tensor::INT64;
  label_tensor.shape = MakeTestNonCpuBuffer<int64_t>(std::vector<int64_t>{1, 300});
  label_tensor.data = MakeTestNonCpuBuffer(BytesOf(labels));
  Tensor box_tensor;
  box_tensor.name = "boxes";
  box_tensor.data_type = Tensor::FLOAT32;
  box_tensor.shape = MakeTestNonCpuBuffer<int64_t>(std::vector<int64_t>{1, 300, 4});
  box_tensor.data = MakeTestNonCpuBuffer(BytesOf(boxes));
  Tensor score_tensor;
  score_tensor.name = "scores";
  score_tensor.data_type = Tensor::FLOAT32;
  score_tensor.shape = MakeTestNonCpuBuffer<int64_t>(std::vector<int64_t>{1, 300});
  score_tensor.data = MakeTestNonCpuBuffer(BytesOf(scores));

  TensorBundle message;
  message.header.frame_id = "camera";
  message.tensors.push_back(std::move(label_tensor));
  message.tensors.push_back(std::move(box_tensor));
  message.tensors.push_back(std::move(score_tensor));
  auto config = Config();

  const auto output = gpu_ros::rtdetr::DecodeRtDetrTensorBundle(message, config);

  ASSERT_EQ(output.detections.size(), 1U);
  EXPECT_EQ(output.detections.front().results.front().hypothesis.class_id, "7");
  EXPECT_FLOAT_EQ(output.detections.front().results.front().hypothesis.score, 0.9F);
}

TEST(RtDetrDecoderTest, DecodesValidBoxesAndFiltersScores)
{
  const std::vector<int64_t> labels{3, 7};
  const std::vector<float> boxes{10.0F, 20.0F, 50.0F, 80.0F, 0.0F, 0.0F, 1.0F, 1.0F};
  const std::vector<float> scores{0.9F, 0.5F};
  std_msgs::msg::Header header;
  header.frame_id = "camera";

  const auto output = DecodeRtDetrValues(header, labels.data(), labels.size(), boxes.data(),
    boxes.size(), scores.data(), scores.size(), Config());

  ASSERT_EQ(output.detections.size(), 1U);
  const auto & detection = output.detections.front();
  EXPECT_EQ(detection.results.front().hypothesis.class_id, "3");
  EXPECT_FLOAT_EQ(detection.results.front().hypothesis.score, 0.9F);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.x, 30.0);
  EXPECT_DOUBLE_EQ(detection.bbox.center.position.y, 50.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_x, 40.0);
  EXPECT_DOUBLE_EQ(detection.bbox.size_y, 60.0);
}

TEST(RtDetrDecoderTest, RejectsInconsistentSpansAndNonFiniteBoxes)
{
  const std::vector<int64_t> labels{1};
  const std::vector<float> boxes{0.0F, 0.0F, 1.0F, 1.0F};
  const std::vector<float> scores{0.9F};
  EXPECT_THROW(DecodeRtDetrValues(std_msgs::msg::Header{}, labels.data(), labels.size(),
                 boxes.data(), 3U, scores.data(), scores.size(), Config()),
    std::invalid_argument);

  const std::vector<float> nonfinite_boxes{
    0.0F, 0.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F};
  const auto output = DecodeRtDetrValues(std_msgs::msg::Header{}, labels.data(), labels.size(),
    nonfinite_boxes.data(), nonfinite_boxes.size(), scores.data(), scores.size(), Config());
  EXPECT_TRUE(output.detections.empty());
}
