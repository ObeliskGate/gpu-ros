#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "gpu_ros_managed_core/detail/backend_ops.hpp"
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
    "input", TensorDataType::kFloat32, {1, 4}, values.data(), values.size() * sizeof(float)));
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
  gpu_ros_managed::ManagedSubscriber<ManagedTensorBundleView> subscriber(subscriber_node.get(),
    "managed_probe",
    [&received](ManagedTensorBundleView view) { received.set_value(view.owner().get()); });
  gpu_ros_managed::ManagedPublisher<ManagedTensorBundle> publisher(
    publisher_node.get(), "managed_probe");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(publisher_node);
  executor.add_node(subscriber_node);
  std::thread spin([&executor] { executor.spin(); });
  std::this_thread::sleep_for(100ms);
  auto message = std::make_unique<ManagedTensorBundle>(host_message());
  const auto * identity = message.get();
  publisher.publish(std::move(message));
  ASSERT_EQ(future.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(future.get(), identity);
  executor.cancel();
  spin.join();
}

namespace
{
class ReservationOps final : public gpu_ros_managed::detail::BackendOps
{
public:
  gpu_ros_managed::BackendKind kind() const noexcept override
  {
    return gpu_ros_managed::BackendKind::kCuda;
  }
  void select_device(int ordinal) override
  {
    if (ordinal != 0) {
      throw std::invalid_argument("Unexpected device");
    }
  }
  gpu_ros_managed::detail::Event create_event() override
  {
    throw std::logic_error("Canceled reservation must not create an event");
  }
  void record_event(gpu_ros_managed::detail::Event,
    gpu_ros_managed::detail::NativeStream) override
  {
    throw std::logic_error("Canceled reservation must not record an event");
  }
  void wait_event(gpu_ros_managed::detail::NativeStream,
    gpu_ros_managed::detail::Event) override
  {
    throw std::logic_error("Canceled reservation must not wait on an event");
  }
  void synchronize_event(gpu_ros_managed::detail::Event) override
  {
    throw std::logic_error("Canceled reservation must not synchronize an event");
  }
  void destroy_event(gpu_ros_managed::detail::Event) noexcept override
  {
    ADD_FAILURE() << "Canceled reservation must not destroy an event";
  }
  std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) override
  {
    select_device(ordinal);
    return std::shared_ptr<void>(new uint8_t[bytes],
      [](void * data) { delete[] static_cast<uint8_t *>(data); });
  }
  void copy_host_to_device(int, void * destination, const void * source, size_t bytes) override
  {
    std::memcpy(destination, source, bytes);
  }
  void copy_device_to_host(int, void * destination, const void * source, size_t bytes) override
  {
    std::memcpy(destination, source, bytes);
  }
};
} // namespace

TEST(TypeAdapterProbe, InvalidPooledTensorReturnsUnsubmittedReservation)
{
  namespace grm = gpu_ros_managed;
  auto ops = std::make_shared<ReservationOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto stream = grm::detail::StreamAccess::make(device, 1, {}, ops);
  auto pool = grm::detail::PoolFactory::make(device, 4, 1, ops);
  const std::vector<std::vector<int64_t>> invalid_shapes{{2}, {}, {0}, {-1}};
  for (const auto & shape : invalid_shapes) {
    EXPECT_THROW(
      grm::tensor_from_pool("input", TensorDataType::kFloat32, shape, pool, stream),
      std::invalid_argument);
    ASSERT_EQ(pool.available(), 1U);
  }
  EXPECT_THROW(
    grm::tensor_from_pool("input", static_cast<TensorDataType>(255), {1}, pool, stream),
    std::invalid_argument);
  ASSERT_EQ(pool.available(), 1U);
  EXPECT_THROW(
    grm::tensor_from_pool("input", TensorDataType::kFloat32,
      {std::numeric_limits<int64_t>::max(), 3}, pool, stream),
    std::overflow_error);
  ASSERT_EQ(pool.available(), 1U);
  EXPECT_THROW(
    grm::tensor_from_pool("input", TensorDataType::kFloat32,
      {std::numeric_limits<int64_t>::max()}, pool, stream),
    std::overflow_error);
  ASSERT_EQ(pool.available(), 1U);
  for (int attempt = 0; attempt < 2; ++attempt) {
    {
      auto tensor = grm::tensor_from_pool(
        "input", TensorDataType::kFloat32, {1}, pool, stream);
      EXPECT_EQ(tensor.tensor.byte_size(), sizeof(float));
      tensor.writer.cancel();
      EXPECT_EQ(pool.available(), 0U);
    }
    EXPECT_EQ(pool.available(), 1U);
  }
  EXPECT_TRUE(pool.shutdown(1s));
}
