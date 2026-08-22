// Copyright 2026 Boshen Chen
// Licensed under the Apache License, Version 2.0.
#include "gpu_ros_managed_hip/hip_backend.hpp"
#include <sstream>
#include <stdexcept>
#include <utility>
#include "gpu_ros_managed_core/detail/backend_ops.hpp"
namespace gpu_ros_managed::hip
{
namespace
{
void check(hipError_t result, const char * operation)
{
  if (result == hipSuccess) {return;}
  std::ostringstream out;
  out << operation << ": " << hipGetErrorName(result) << " (" << hipGetErrorString(result) << ")";
  throw std::runtime_error(out.str());
}
class HipOps final : public detail::BackendOps
{
public:
  BackendKind kind() const noexcept override {return BackendKind::kHip;}
  void select_device(int id) override {check(hipSetDevice(id), "hipSetDevice");}
  detail::Event create_event() override
  {
    hipEvent_t event{};
    check(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreateWithFlags");
    return reinterpret_cast<detail::Event>(event);
  }
  void record_event(detail::Event e, detail::NativeStream s) override
  {check(hipEventRecord(reinterpret_cast<hipEvent_t>(e), reinterpret_cast<hipStream_t>(s)), "hipEventRecord");}
  void wait_event(detail::NativeStream s, detail::Event e) override
  {check(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(s), reinterpret_cast<hipEvent_t>(e), 0), "hipStreamWaitEvent");}
  void synchronize_event(detail::Event e) override
  {check(hipEventSynchronize(reinterpret_cast<hipEvent_t>(e)), "hipEventSynchronize");}
  void destroy_event(detail::Event e) noexcept override
  {if (e) {static_cast<void>(hipEventDestroy(reinterpret_cast<hipEvent_t>(e)));}}
  std::shared_ptr<void> allocate_device(int id, size_t bytes) override
  {
    select_device(id);
    void * pointer{};
    check(hipMalloc(&pointer, bytes), "hipMalloc");
    return std::shared_ptr<void>(pointer, [id](void * p) {
      static_cast<void>(hipSetDevice(id)); static_cast<void>(hipFree(p));
    });
  }
  void copy_host_to_device(int id, void * dst, const void * src, size_t bytes) override
  {select_device(id); check(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");}
  void copy_device_to_host(int id, void * dst, const void * src, size_t bytes) override
  {select_device(id); check(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");}
};
std::shared_ptr<detail::BackendOps> ops()
{
  static auto value = std::make_shared<HipOps>();
  return value;
}
}  // namespace
hipStream_t HipStream::get() const
{return reinterpret_cast<hipStream_t>(detail::StreamAccess::get(stream_).native);}
HipStream make_stream(int id)
{
  ops()->select_device(id);
  hipStream_t native{};
  check(hipStreamCreateWithFlags(&native, hipStreamNonBlocking), "hipStreamCreateWithFlags");
  auto owner = std::shared_ptr<void>(reinterpret_cast<void *>(native), [id](void * p) {
    static_cast<void>(hipSetDevice(id)); static_cast<void>(hipStreamDestroy(reinterpret_cast<hipStream_t>(p)));
  });
  return HipStream(detail::DeviceBufferFactory::make_stream(
      {BackendKind::kHip, id}, reinterpret_cast<detail::NativeStream>(native), std::move(owner), ops()));
}
HipStream wrap_borrowed_stream(hipStream_t native, int id, std::shared_ptr<void> owner)
{
  if (!native) {throw std::invalid_argument("Borrowed HIP stream is null");}
  return HipStream(detail::DeviceBufferFactory::make_stream(
      {BackendKind::kHip, id}, reinterpret_cast<detail::NativeStream>(native), std::move(owner), ops()));
}
std::shared_ptr<DeviceBuffer> allocate(size_t bytes, int id)
{
  auto owner = ops()->allocate_device(id, bytes);
  void * pointer = owner.get();
  return detail::DeviceBufferFactory::make_fresh(
    {BackendKind::kHip, id}, pointer, bytes, std::move(owner), ops());
}
std::shared_ptr<DeviceBuffer> adopt_external(void * p, size_t n, int id, std::shared_ptr<void> owner)
{return detail::DeviceBufferFactory::make_fresh({BackendKind::kHip, id}, p, n, std::move(owner), ops());}
std::shared_ptr<DeviceBuffer> adopt_synchronized_external(void * p, size_t n, int id, std::shared_ptr<void> owner)
{return detail::DeviceBufferFactory::make_ready({BackendKind::kHip, id}, p, n, std::move(owner), ops());}
FixedDeviceMemoryPool make_fixed_device_pool(size_t n, size_t count, int id)
{return detail::PoolFactory::make({BackendKind::kHip, id}, n, count, ops());}
bool wait_for_pending_releases(std::chrono::milliseconds timeout)
{return detail::wait_for_pending_releases(BackendKind::kHip, timeout);}
}  // namespace gpu_ros_managed::hip
