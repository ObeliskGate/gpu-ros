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
#include "gpu_ros_onnx_inference/managed_tensor_bundle_adapter.hpp"
#include "gpu_ros_onnx_inference/rosidl_buffer_adapter.hpp"
#include "gpu_ros_onnx_inference/tensor_bundle_io.hpp"
#include "gpu_ros_onnx_inference/tensor_dtype.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"
#include "gpu_ros_managed_ros/managed_pub_sub.hpp"
#include "gpu_ros_managed_tensor_bundle/type_adapter.hpp"
#include <stdexcept>
#include <utility>
#ifdef GPU_ROS_MANAGED_CUDA
#include "gpu_ros_managed_cuda/cuda_backend.hpp"
#endif
#ifdef GPU_ROS_MANAGED_HIP
#include "gpu_ros_managed_hip/hip_backend.hpp"
#endif
#ifdef GPU_ROS_ORT_MANAGED_TEST
#include "onnx_binding_test_peer.hpp"
#endif
namespace gpu_ros::onnx_inference
{
TensorBindingBatch BindManagedTensorBundle(gpu_ros_managed::ManagedTensorBundleView input,
  ExecutionProvider provider, int device, bool strict)
{
  TensorBindingBatch batch{input.header(), input.owner(), {}};
  batch.bindings.reserve(input.tensors().size());
  for (const auto & tensor : input.tensors()) {
    std::shared_ptr<const void> owner = input.owner();
    const void * pointer;
    auto memory = ProviderMemoryInfo(ExecutionProvider::kCpu, device);
    std::string storage = "host";
    if (const auto * host = std::get_if<gpu_ros_managed::HostBuffer>(&tensor.storage())) {
      if (strict) { throw std::invalid_argument("device_strict does not accept host Managed tensors"); }
      pointer = host->data();
    } else {
      const auto & buffer = std::get<std::shared_ptr<gpu_ros_managed::DeviceBuffer>>(tensor.storage());
      if (!buffer || buffer->device_id().ordinal != device) {
        throw std::invalid_argument("Managed input device mismatch");
      }
      const bool cuda = buffer->device_id().backend == gpu_ros_managed::BackendKind::kCuda;
      const bool hip = buffer->device_id().backend == gpu_ros_managed::BackendKind::kHip;
      if ((!cuda || provider != ExecutionProvider::kCuda) &&
        (!hip || (provider != ExecutionProvider::kMigraphx && provider != ExecutionProvider::kRocm)))
      { throw std::invalid_argument("Managed input backend/provider mismatch"); }
      auto lease = std::make_shared<gpu_ros_managed::BlockingReadyLease>(buffer->get_blocking_ready_lease());
      pointer = lease->data();
      owner = std::move(lease);
      memory = ProviderMemoryInfo(provider, device);
      storage = cuda ? "cuda_device" : "hip_device";
    }
    const auto dtype = BundleToOnnxDtype(static_cast<uint8_t>(tensor.data_type()));
    const auto bytes = TensorByteSize(dtype, tensor.shape(), tensor.name());
    if (bytes != tensor.byte_size()) { throw std::invalid_argument("Managed input byte size mismatch"); }
    batch.bindings.push_back({tensor.name(), std::move(owner), pointer, bytes, std::move(storage),
      "Managed ready lease retained through output synchronization",
      Ort::Value::CreateTensor(memory, const_cast<void *>(pointer), bytes,
        tensor.shape().data(), tensor.shape().size(), dtype)});
  }
  return batch;
}
namespace
{
using Buffer = gpu_ros_managed::DeviceBuffer;
using Reservation = gpu_ros_managed::SynchronizedPoolBlock;
struct ManagedState
{
  std::shared_ptr<const void> retained;
  std::vector<std::unique_ptr<Reservation>> reservations;
  bool terminal{false};
};
class ManagedBatch final : public DeviceOutputBatch
{
public:
  explicit ManagedBatch(size_t count) : state_(std::make_shared<ManagedState>())
  {
    // Reserve before acquiring writers: Add must not throw and terminalize an
    // unsubmitted reservation while growing its owner container.
    state_->reservations.reserve(count);
  }
  ~ManagedBatch() override { if (!state_->terminal) { FailAfterSubmit(); } }
  void Add(std::unique_ptr<Reservation> reservation) { state_->reservations.push_back(std::move(reservation)); }
  size_t size() const noexcept override { return state_->reservations.size(); }
  void * pointer(size_t i) const override { return state_->reservations.at(i)->writer.data(); }
  TensorStorage storage(size_t i) const override
  {
    const auto & buffer = state_->reservations.at(i)->buffer;
    return {buffer, buffer, pointer(i), buffer->size(),
      buffer->device_id().backend == gpu_ros_managed::BackendKind::kCuda ? "cuda_device" : "hip_device"};
  }
  void RetainOwner(std::shared_ptr<const void> owner) override { state_->retained = std::move(owner); }
  void CompleteAfterSync() override
  {
    if (state_->terminal) { throw std::logic_error("Managed transaction already terminal"); }
    for (auto & block : state_->reservations) { block->writer.finalize_synchronously(); }
    state_->terminal = true;
    state_->retained.reset();
  }
  void CancelBeforeSubmit() noexcept override
  {
    if (state_->terminal) { return; }
    for (auto & block : state_->reservations) {
      try { block->writer.cancel(); } catch (...) { block->writer.fail(); }
    }
    state_->terminal = true;
    state_->retained.reset();
  }
  void FailAfterSubmit() noexcept override
  {
    if (state_->terminal) { return; }
    for (auto & block : state_->reservations) { block->writer.fail(); }
    state_->terminal = true;
  }
  void CopyFrom(const std::vector<OutputTensor> &) override
  { throw std::logic_error("Managed dynamic outputs must be adopted, not copied"); }
private:
  std::shared_ptr<ManagedState> state_;
};
class ManagedAllocator final : public DeviceOutputAllocator
{
public:
  explicit ManagedAllocator(const OnnxInferenceCore::Config & cfg) : config_(cfg) {}
  ~ManagedAllocator() override { static_cast<void>(shutdown(std::chrono::seconds(5))); }
  std::unique_ptr<DeviceOutputBatch> Allocate(const std_msgs::msg::Header &,
    const std::vector<DeviceOutputSpec> & specs) override
  {
    const bool strict = config_.io_contract == "device_strict";
    if (strict && pools_.empty()) {
      std::vector<std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool>> pools;
      pools.reserve(specs.size());
      for (const auto & spec : specs) {
        const auto bytes = TensorByteSize(spec.dtype, spec.shape, spec.name);
#ifdef GPU_ROS_MANAGED_CUDA
        if (config_.ep == ExecutionProvider::kCuda) {
          pools.push_back(std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
            gpu_ros_managed::cuda::make_fixed_device_pool(bytes, config_.output_pool_capacity, config_.gpu_device_id)));
          continue;
        }
#endif
#ifdef GPU_ROS_MANAGED_HIP
        pools.push_back(std::make_unique<gpu_ros_managed::FixedDeviceMemoryPool>(
          gpu_ros_managed::hip::make_fixed_device_pool(bytes, config_.output_pool_capacity, config_.gpu_device_id)));
#else
        (void)bytes;
        throw std::runtime_error("Managed backend unavailable");
#endif
      }
      pools_ = std::move(pools);
    }
    auto batch = std::make_unique<ManagedBatch>(specs.size());
    const auto deadline = std::chrono::steady_clock::now() + config_.output_pool_wait_timeout;
    try {
      for (size_t i = 0; i < specs.size(); ++i) {
        std::unique_ptr<Reservation> reservation;
        if (strict) {
          const auto now = std::chrono::steady_clock::now();
          reservation = pools_.at(i)->acquire_synchronized_for(now >= deadline ? std::chrono::milliseconds(0) :
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
          if (!reservation) { throw std::runtime_error("strict Managed output pool exhausted"); }
        } else {
          std::shared_ptr<Buffer> buffer;
          const auto bytes = TensorByteSize(specs[i].dtype, specs[i].shape, specs[i].name);
#ifdef GPU_ROS_MANAGED_CUDA
          if (config_.ep == ExecutionProvider::kCuda) {
            buffer = gpu_ros_managed::cuda::allocate(bytes, config_.gpu_device_id);
          }
#endif
#ifdef GPU_ROS_MANAGED_HIP
          if (config_.ep == ExecutionProvider::kMigraphx || config_.ep == ExecutionProvider::kRocm) {
            buffer = gpu_ros_managed::hip::allocate(bytes, config_.gpu_device_id);
          }
#endif
          (void)bytes;
          if (!buffer) { throw std::runtime_error("Managed backend unavailable"); }
          // A new-expression allocates before evaluating constructor arguments;
          // make_unique would acquire the writer before allocating its wrapper.
          reservation.reset(new Reservation(buffer, buffer->get_synchronized_write_handle()));
        }
        batch->Add(std::move(reservation));
#ifdef GPU_ROS_ORT_MANAGED_TEST
        OnnxBindingTestPeer::Reservation(i, batch->storage(i));
#endif
      }
      return batch;
    } catch (...) { batch->CancelBeforeSubmit(); throw; }
  }
  Ort::MemoryInfo memory_info() const override { return ProviderMemoryInfo(config_.ep, config_.gpu_device_id); }
  bool adopts_dynamic_outputs() const noexcept override { return true; }
  bool permits_preallocation_fallback() const noexcept override
  { return config_.ep == ExecutionProvider::kMigraphx && config_.io_contract.empty(); }
  TensorStorage Adopt(TensorStorage storage) override
  {
    std::shared_ptr<Buffer> buffer;
    auto owner = std::const_pointer_cast<void>(storage.owner);
#ifdef GPU_ROS_MANAGED_CUDA
    if (config_.ep == ExecutionProvider::kCuda) {
      buffer = gpu_ros_managed::cuda::adopt_synchronized_external(storage.data,
        storage.byte_count, config_.gpu_device_id, owner);
    }
#endif
#ifdef GPU_ROS_MANAGED_HIP
    if (config_.ep == ExecutionProvider::kMigraphx || config_.ep == ExecutionProvider::kRocm) {
      buffer = gpu_ros_managed::hip::adopt_synchronized_external(storage.data,
        storage.byte_count, config_.gpu_device_id, owner);
    }
#endif
    if (!buffer) { throw std::runtime_error("Managed backend unavailable"); }
    storage.owner = buffer;
    storage.envelope = buffer;
    return storage;
  }
  bool SynchronizeAfterFailure() noexcept override
  {
#ifdef GPU_ROS_MANAGED_CUDA
    if (config_.ep == ExecutionProvider::kCuda) {
      return cudaSetDevice(config_.gpu_device_id) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
#endif
#ifdef GPU_ROS_MANAGED_HIP
    if (config_.ep == ExecutionProvider::kMigraphx || config_.ep == ExecutionProvider::kRocm) {
      return hipSetDevice(config_.gpu_device_id) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
    }
#endif
    return false;
  }
  bool shutdown(std::chrono::milliseconds timeout) noexcept override
  {
    bool drained = true;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (auto & pool : pools_) {
      const auto now = std::chrono::steady_clock::now();
      try {
        drained = pool->shutdown(now >= deadline ? std::chrono::milliseconds(0) :
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)) && drained;
      } catch (...) { drained = false; }
    }
    return drained;
  }
private:
  OnnxInferenceCore::Config config_;
  std::vector<std::unique_ptr<gpu_ros_managed::FixedDeviceMemoryPool>> pools_;
};
class ManagedIO final : public ITensorBundleIO
{
public:
  ManagedIO(rclcpp::Node * node, bool publish) : node_(node)
  {
    config_.ep = ParseExecutionProvider(node->get_parameter("execution_provider").as_string());
    config_.gpu_device_id = static_cast<int>(node->get_parameter("gpu_device_id").as_int());
    if (node->has_parameter("io_contract")) { config_.io_contract = node->get_parameter("io_contract").as_string(); }
    if (node->has_parameter("output_pool_capacity")) {
      const auto count = node->get_parameter("output_pool_capacity").as_int();
      if (count <= 0) { throw std::invalid_argument("output pool capacity must be positive"); }
      config_.output_pool_capacity = static_cast<size_t>(count);
    }
    if (node->has_parameter("output_pool_wait_timeout_ms")) {
      config_.output_pool_wait_timeout = std::chrono::milliseconds(node->get_parameter("output_pool_wait_timeout_ms").as_int());
    }
    if (config_.ep == ExecutionProvider::kCuda || config_.ep == ExecutionProvider::kMigraphx) {
      allocator_ = CreateManagedOutputAllocator(config_);
    }
    if (publish) { publisher_ = std::make_unique<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>>(
      node, "tensor_output", rclcpp::QoS(10)); }
  }
  void Subscribe(Callback callback) override
  {
    subscriber_ = std::make_unique<gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>>(
      node_, "tensor_input", [callback = std::move(callback), cfg = config_](gpu_ros_managed::ManagedTensorBundleView input) {
        callback(BindManagedTensorBundle(std::move(input), cfg.ep, cfg.gpu_device_id, cfg.io_contract == "device_strict"));
      }, rclcpp::QoS(10));
  }
  OutputPlacement output_placement() const noexcept override
  { return allocator_ ? OutputPlacement::kDevice : OutputPlacement::kHost; }
  DeviceOutputAllocator * device_output_allocator() noexcept override { return allocator_.get(); }
  void Publish(TensorBundleOutput && output) override
  {
    if (!publisher_) { throw std::logic_error("Managed publisher disabled"); }
    publisher_->publish(ManagedOutputMessage(std::move(output)));
  }
private:
  rclcpp::Node * node_;
  OnnxInferenceCore::Config config_;
  std::unique_ptr<DeviceOutputAllocator> allocator_;
  std::unique_ptr<gpu_ros_managed::ManagedPublisher<gpu_ros_managed::ManagedTensorBundle>> publisher_;
  std::unique_ptr<gpu_ros_managed::ManagedSubscriber<gpu_ros_managed::ManagedTensorBundleView>> subscriber_;
};
} // namespace
std::unique_ptr<DeviceOutputAllocator> CreateManagedOutputAllocator(const OnnxInferenceCore::Config & config)
{ return std::make_unique<ManagedAllocator>(config); }
gpu_ros_managed::ManagedTensorBundle ManagedOutputMessage(TensorBundleOutput output)
{
  std::vector<gpu_ros_managed::ManagedTensor> tensors;
  tensors.reserve(output.tensors.size());
  for (auto & tensor : output.tensors) {
    const auto dtype = static_cast<gpu_ros_managed::TensorDataType>(OnnxToBundleDtype(tensor.dtype));
    if (auto * host = std::get_if<std::vector<uint8_t>>(&tensor.storage)) {
      auto owner = std::make_shared<std::vector<uint8_t>>(std::move(*host));
      tensors.push_back(gpu_ros_managed::ManagedTensor::from_host_external(std::move(tensor.name),
        dtype, std::move(tensor.shape), owner, owner->data(), owner->size()));
    } else {
      auto buffer = std::static_pointer_cast<Buffer>(std::get<TensorStorage>(tensor.storage).envelope);
      if (!buffer) { throw std::invalid_argument("Managed output has no storage envelope"); }
      tensors.emplace_back(std::move(tensor.name), dtype, std::move(tensor.shape), std::move(buffer));
    }
  }
  return {std::move(output.header), std::move(tensors)};
}
std::unique_ptr<ITensorBundleIO> CreateManagedTensorBundleIO(rclcpp::Node * node, bool publish)
{ return std::make_unique<ManagedIO>(node, publish); }
} // namespace gpu_ros::onnx_inference
