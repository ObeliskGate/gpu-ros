#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/tensor_bundle.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include "gpu_ros_tensor_bundle_msgs/msg/tensor.hpp"

using namespace std::chrono_literals;
using gpu_ros_managed::ManagedTensor;
using gpu_ros_managed::ManagedTensorBundle;
using gpu_ros_managed::ManagedTensorBundleView;
using gpu_ros_managed::TensorDataType;
using RosTensorBundle = gpu_ros_tensor_bundle_msgs::msg::TensorBundle;
using Adapter = rclcpp::TypeAdapter<ManagedTensorBundle, RosTensorBundle>;

ManagedTensorBundle host_message()
{
  const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<ManagedTensor> tensors;
  tensors.push_back(ManagedTensor::from_host_copy(
      "input", TensorDataType::kFloat32, {1, 4},
      values.data(), values.size() * sizeof(float)));
  return ManagedTensorBundle(std_msgs::msg::Header{}, std::move(tensors));
}

TEST(TypeAdapterProbe, ManagedToRosFallbackPreservesValuesAndShape)
{
  RosTensorBundle ros;
  Adapter::convert_to_ros_message(host_message(), ros);
  ASSERT_EQ(ros.tensors.size(), 1U);
  EXPECT_EQ(ros.tensors[0].data_type, gpu_ros_tensor_bundle_msgs::msg::Tensor::FLOAT32);
  EXPECT_EQ(ros.tensors[0].shape, (std::vector<int64_t>{1, 4}));
  ASSERT_EQ(ros.tensors[0].data.size(), 4U * sizeof(float));
  const auto * values = reinterpret_cast<const float *>(ros.tensors[0].data.data());
  EXPECT_FLOAT_EQ(values[0], 1.0F);
  EXPECT_FLOAT_EQ(values[3], 4.0F);
}

TEST(TypeAdapterProbe, RosToManagedFallbackProducesHostBuffer)
{
  RosTensorBundle ros;
  Adapter::convert_to_ros_message(host_message(), ros);
  ManagedTensorBundle managed(std_msgs::msg::Header{}, {});
  Adapter::convert_to_custom(ros, managed);
  ASSERT_EQ(managed.tensors().size(), 1U);
  EXPECT_TRUE(managed.tensors()[0].is_host());
  EXPECT_EQ(managed.tensors()[0].data_type(), TensorDataType::kFloat32);
  EXPECT_EQ(managed.tensors()[0].shape(), (std::vector<int64_t>{1, 4}));
}

TEST(TypeAdapterProbe, NativeIntraProcessPreservesCustomObjectIdentity)
{
  if (!rclcpp::ok()) {
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  auto publisher_node = std::make_shared<rclcpp::Node>("managed_probe_pub", options);
  auto subscriber_node = std::make_shared<rclcpp::Node>("managed_probe_sub", options);
  std::promise<const ManagedTensorBundle *> received;
  auto future = received.get_future();
  gpu_ros_managed::ManagedSubscriber<ManagedTensorBundleView> subscriber(
    subscriber_node.get(), "managed_probe",
    [&received](ManagedTensorBundleView view) {received.set_value(view.owner().get());});
  gpu_ros_managed::ManagedPublisher<ManagedTensorBundle> publisher(
    publisher_node.get(), "managed_probe");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(publisher_node);
  executor.add_node(subscriber_node);
  std::thread spin([&executor] {executor.spin();});
  std::this_thread::sleep_for(100ms);
  auto message = std::make_unique<ManagedTensorBundle>(host_message());
  const auto * identity = message.get();
  publisher.publish(std::move(message));
  ASSERT_EQ(future.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(future.get(), identity);
  executor.cancel();
  spin.join();
}
