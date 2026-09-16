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

#include <limits>
#include <stdexcept>

#include "gpu_ros_onnx_inference/staging_pool_limits.hpp"

namespace detail = gpu_ros::onnx_inference::detail;

TEST(StagingPoolLimits, AcceptsBoundedAddition)
{
  EXPECT_EQ(detail::ValidateStagingPoolAddition(64, 4, 100, 1, 64, 1000, 2), 356U);
}

TEST(StagingPoolLimits, RejectsSingleTensorLimit)
{
  EXPECT_THROW(detail::ValidateStagingPoolAddition(65, 1, 0, 0, 64, 1000, 2), std::runtime_error);
}

TEST(StagingPoolLimits, RejectsEntryLimit)
{
  EXPECT_THROW(detail::ValidateStagingPoolAddition(1, 1, 0, 2, 64, 1000, 2), std::runtime_error);
}

TEST(StagingPoolLimits, RejectsTotalByteLimit)
{
  EXPECT_THROW(detail::ValidateStagingPoolAddition(64, 4, 800, 1, 64, 1000, 2), std::runtime_error);
}

TEST(StagingPoolLimits, RejectsMultiplicationOverflow)
{
  EXPECT_THROW(detail::ValidateStagingPoolAddition(std::numeric_limits<size_t>::max(), 2, 0, 0,
                 std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), 2),
    std::overflow_error);
}

TEST(StagingPoolLimits, RejectsAdditionOverflow)
{
  EXPECT_THROW(detail::ValidateStagingPoolAddition(1, 1, std::numeric_limits<size_t>::max(), 0, 1,
                 std::numeric_limits<size_t>::max(), 2),
    std::overflow_error);
}
