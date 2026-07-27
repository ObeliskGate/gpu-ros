#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_managed_core/detail/backend_ops.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"

namespace grm = gpu_ros_managed;

class FakeOps final : public grm::detail::BackendOps
{
public:
  grm::BackendKind kind() const noexcept override {return grm::BackendKind::kCuda;}
  void select_device(int ordinal) override {selected = ordinal;}
  grm::detail::Event create_event() override
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto event = ++next_event;
    live.insert(event);
    return event;
  }
  void record_event(grm::detail::Event event, grm::detail::NativeStream stream) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!live.count(event)) {throw std::runtime_error("unknown event");}
    last_record_stream = stream;
    ++records;
  }
  void wait_event(grm::detail::NativeStream stream, grm::detail::Event event) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!live.count(event)) {throw std::runtime_error("unknown event");}
    last_wait_stream = stream;
    ++waits;
  }
  void synchronize_event(grm::detail::Event event) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!live.count(event)) {throw std::runtime_error("unknown event");}
    ++synchronizes;
  }
  void destroy_event(grm::detail::Event event) noexcept override
  {
    std::lock_guard<std::mutex> lock(mutex);
    live.erase(event);
    ++destroys;
  }
  std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) override
  {
    selected = ordinal;
    return std::shared_ptr<void>(std::malloc(bytes), [](void * p) {std::free(p);});
  }
  void copy_device_to_host(int, void *, const void *, size_t) override {}

  std::atomic<int> records{0};
  std::atomic<int> waits{0};
  std::atomic<int> synchronizes{0};
  std::atomic<int> destroys{0};
  int selected{-1};
  grm::detail::NativeStream last_record_stream{0};
  grm::detail::NativeStream last_wait_stream{0};
  grm::detail::Event next_event{0};
  std::unordered_set<grm::detail::Event> live;
  std::mutex mutex;
};

template<typename Exception, typename Function>
void expect_throws(Function && function)
{
  bool threw = false;
  try {function();} catch (const Exception &) {threw = true;}
  assert(threw);
}

int main()
{
  static_assert(std::is_move_constructible_v<grm::WriteHandle>);
  static_assert(!std::is_move_assignable_v<grm::WriteHandle>);
  static_assert(std::is_move_constructible_v<grm::ReadHandle>);
  static_assert(!std::is_move_assignable_v<grm::ReadHandle>);
  static_assert(!std::is_move_assignable_v<grm::BlockingReadyLease>);
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  {
    const grm::DeviceId device{grm::BackendKind::kCuda, 2};
    auto producer = grm::detail::DeviceBufferFactory::make_stream(device, 11, {}, ops);
    auto consumer_a = grm::detail::DeviceBufferFactory::make_stream(device, 21, {}, ops);
    auto consumer_b = grm::detail::DeviceBufferFactory::make_stream(device, 22, {}, ops);
    auto wrong_device = grm::detail::DeviceBufferFactory::make_stream(
      {grm::BackendKind::kCuda, 3}, 31, {}, ops);

    auto allocation = ops->allocate_device(2, 64);
    expect_throws<std::invalid_argument>([&] {
      grm::detail::DeviceBufferFactory::make_fresh(
        device, allocation.get(), 64, {}, ops);
    });
    auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
      device, allocation.get(), 64, allocation, ops);
    auto writer = buffer->get_write_handle(producer);
    assert(writer.data() == allocation.get());
    expect_throws<std::logic_error>([&] {buffer->get_read_handle(consumer_a);});
    expect_throws<std::logic_error>([&] {buffer->get_write_handle(producer);});
    writer.finalize();
    writer.finalize();
    assert(ops->last_record_stream == 11);
    expect_throws<std::invalid_argument>([&] {buffer->get_read_handle(wrong_device);});

    auto reader_a = buffer->get_read_handle(consumer_a);
    auto reader_b = buffer->get_read_handle(consumer_b);
    assert(ops->waits == 2);
    assert(reader_a.data() == allocation.get());
    reader_a.finish();
    reader_a.finish();
    reader_b.finish();
    assert(ops->records == 3);
    expect_throws<std::logic_error>([&] {buffer->get_write_handle(producer);});

    auto lease = buffer->get_blocking_ready_lease();
    assert(lease.data() == allocation.get());
    assert(ops->synchronizes == 1);

    {
      auto raw = std::shared_ptr<void>(
        std::malloc(8), [&owner_releases](void * p) {
          ++owner_releases;
          std::free(p);
        });
      auto fallback = grm::detail::DeviceBufferFactory::make_fresh(
        device, raw.get(), 8, raw, ops);
      {
        auto fallback_writer = fallback->get_write_handle(producer);
        assert(fallback_writer.data() != nullptr);
      }
      auto retained_reader = fallback->get_read_handle(consumer_a);
      fallback.reset();
      raw.reset();
      assert(owner_releases == 0);
      retained_reader.finish();
    }

    auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
    auto held = std::make_unique<grm::PoolBlock>(pool.acquire(producer));
    held->writer.finalize();
    assert(pool.available() == 0);
    assert(pool.acquire_for(producer, std::chrono::milliseconds(1)) == nullptr);
    held.reset();
    auto recycled = pool.acquire_for(producer, std::chrono::seconds(1));
    assert(recycled != nullptr);
    recycled->writer.finalize();
    recycled.reset();
  }
  assert(grm::detail::wait_for_pending_releases(
      grm::BackendKind::kCuda, std::chrono::seconds(2)));
  assert(owner_releases == 1);
  return 0;
}
