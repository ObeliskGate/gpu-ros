#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gpu_ros_managed_core/buffer.hpp"
#include "gpu_ros_managed_core/detail/backend_ops.hpp"
#include "gpu_ros_managed_core/fixed_device_memory_pool.hpp"

namespace grm = gpu_ros_managed;
using namespace std::chrono_literals;

class FakeOps final : public grm::detail::BackendOps
{
public:
  grm::BackendKind kind() const noexcept override { return grm::BackendKind::kCuda; }

  void select_device(int ordinal) override
  {
    if (fail_select.exchange(false)) {
      throw std::runtime_error("select failure");
    }
    selected = ordinal;
    ++select_calls;
  }

  grm::detail::Event create_event() override
  {
    check_event_device();
    if (fail_create.exchange(false)) {
      throw std::runtime_error("create failure");
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto event = ++next_event;
    live.insert(event);
    return event;
  }

  void record_event(grm::detail::Event event, grm::detail::NativeStream stream) override
  {
    check_event_device();
    std::lock_guard<std::mutex> lock(mutex);
    require_live(event);
    last_record_stream = stream;
    ++records;
    if (fail_record.exchange(false)) {
      throw std::runtime_error("record failure");
    }
  }

  void wait_event(grm::detail::NativeStream stream, grm::detail::Event event) override
  {
    check_event_device();
    std::lock_guard<std::mutex> lock(mutex);
    require_live(event);
    last_wait_stream = stream;
    ++waits;
    if (fail_wait.exchange(false)) {
      throw std::runtime_error("wait failure");
    }
  }

  void synchronize_event(grm::detail::Event event) override
  {
    check_event_device();
    std::unique_lock<std::mutex> lock(mutex);
    require_live(event);
    ++synchronizes;
    synchronize_entered = true;
    synchronize_entered_cv.notify_all();
    synchronize_gate_cv.wait(lock, [&] { return allow_synchronize; });
    if (fail_synchronize_events.erase(event) != 0) {
      throw std::runtime_error("synchronize failure");
    }
  }

  void destroy_event(grm::detail::Event event) noexcept override
  {
    check_event_device();
    std::lock_guard<std::mutex> lock(mutex);
    live.erase(event);
    ++destroy_counts[event];
    ++destroys;
  }

  std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) override
  {
    selected = ordinal;
    void * pointer = std::malloc(bytes);
    if (pointer == nullptr) {
      throw std::bad_alloc();
    }
    return std::shared_ptr<void>(pointer, [this](void * value) {
      ++allocation_releases;
      std::free(value);
    });
  }

  void copy_host_to_device(int, void * destination, const void * source, size_t bytes) override
  {
    ++h2d_copies;
    if (fail_h2d.exchange(false)) {
      throw std::runtime_error("H2D failure");
    }
    if (bytes != 0) {
      std::memcpy(destination, source, bytes);
    }
  }

  void copy_device_to_host(int, void * destination, const void * source, size_t bytes) override
  {
    ++d2h_copies;
    if (bytes != 0) {
      std::memcpy(destination, source, bytes);
    }
  }

  void fail_synchronize(grm::detail::Event event)
  {
    std::lock_guard<std::mutex> lock(mutex);
    fail_synchronize_events.insert(event);
  }

  int destroy_count(grm::detail::Event event)
  {
    std::lock_guard<std::mutex> lock(mutex);
    return destroy_counts[event];
  }

  bool wait_until_synchronize_entered(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex);
    return synchronize_entered_cv.wait_for(lock, timeout, [&] { return synchronize_entered; });
  }

  void unblock_synchronize()
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      allow_synchronize = true;
    }
    synchronize_gate_cv.notify_all();
  }

  std::atomic<bool> fail_select{false};
  std::atomic<bool> fail_create{false};
  std::atomic<bool> fail_record{false};
  std::atomic<bool> fail_wait{false};
  std::atomic<int> records{0};
  std::atomic<int> waits{0};
  std::atomic<int> synchronizes{0};
  std::atomic<int> destroys{0};
  std::atomic<int> allocation_releases{0};
  std::atomic<int> h2d_copies{0};
  std::atomic<int> d2h_copies{0};
  std::atomic<bool> fail_h2d{false};
  std::atomic<int> selected{-1};
  std::atomic<int> expected_event_device{0};
  std::atomic<int> select_calls{0};
  std::atomic<int> event_device_failures{0};
  grm::detail::NativeStream last_record_stream{0};
  grm::detail::NativeStream last_wait_stream{0};
  bool allow_synchronize{true};

private:
  void check_event_device() noexcept
  {
    if (selected.exchange(-1) != expected_event_device.load()) {
      ++event_device_failures;
    }
  }

  void require_live(grm::detail::Event event)
  {
    if (live.count(event) == 0) {
      throw std::runtime_error("unknown event");
    }
  }

  grm::detail::Event next_event{0};
  std::unordered_set<grm::detail::Event> live;
  std::unordered_set<grm::detail::Event> fail_synchronize_events;
  std::unordered_map<grm::detail::Event, int> destroy_counts;
  std::mutex mutex;
  std::condition_variable synchronize_entered_cv;
  std::condition_variable synchronize_gate_cv;
  bool synchronize_entered{false};
};

template <typename Exception, typename Function> void expect_throws(Function && function)
{
  bool threw = false;
  try {
    function();
  } catch (const Exception &) {
    threw = true;
  }
  assert(threw);
}

struct OwnedMemory
{
  void * pointer;
  std::shared_ptr<void> owner;
};

OwnedMemory make_owned_memory(std::atomic<int> & releases, size_t bytes = 64)
{
  void * pointer = std::malloc(bytes);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return {pointer, std::shared_ptr<void>(pointer, [&releases](void * value) {
            ++releases;
            std::free(value);
          })};
}

grm::DeviceStream make_stream(const grm::DeviceId & device, grm::detail::NativeStream native,
  const std::shared_ptr<FakeOps> & ops)
{
  return grm::detail::DeviceBufferFactory::make_stream(device, native, {}, ops);
}

void wait_for_cleanup()
{
  assert(grm::detail::wait_for_pending_releases(grm::BackendKind::kCuda, 2s));
  assert(grm::detail::pending_release_count(grm::BackendKind::kCuda) == 0);
}

void test_blocking_copy_api()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto memory = make_owned_memory(owner_releases, 16);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 16, memory.owner, ops);
  const uint8_t source[4]{1, 2, 3, 4};

  expect_throws<std::invalid_argument>([&] { buffer->copy_from_host_blocking(nullptr, 4); });
  expect_throws<std::out_of_range>([&] { buffer->copy_from_host_blocking(source, 17); });
  buffer->copy_from_host_blocking(source, sizeof(source));
  expect_throws<std::logic_error>([&] { buffer->copy_from_host_blocking(source, 1); });

  uint8_t destination[4]{};
  buffer->copy_to_host_blocking(destination, sizeof(destination));
  assert(std::memcmp(destination, source, sizeof(source)) == 0);
  assert(ops->h2d_copies == 1);
  assert(ops->d2h_copies == 1);
  buffer.reset();
  memory.owner.reset();

  auto producer = make_stream(device, 17, ops);
  auto produced_memory = make_owned_memory(owner_releases, 16);
  auto produced_buffer = grm::detail::DeviceBufferFactory::make_fresh(
    device, produced_memory.pointer, 16, produced_memory.owner, ops);
  std::memcpy(produced_memory.pointer, source, sizeof(source));
  produced_buffer->get_write_handle(producer).finalize();
  uint8_t produced_destination[4]{};
  produced_buffer->copy_to_host_blocking(produced_destination, sizeof(produced_destination));
  assert(std::memcmp(produced_destination, source, sizeof(source)) == 0);
  assert(ops->synchronizes == 1);
  produced_buffer.reset();
  produced_memory.owner.reset();
  wait_for_cleanup();
  assert(owner_releases == 2);
}

void test_synchronized_owner_releases_without_worker_thread()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto memory = make_owned_memory(owner_releases);
  {
    auto buffer = grm::detail::DeviceBufferFactory::make_ready(
      device, memory.pointer, 64, std::move(memory.owner), ops);
    buffer.reset();
  }
  assert(owner_releases == 1);

  auto failed_ops = std::make_shared<FakeOps>();
  failed_ops->fail_select = true;
  std::atomic<int> orphaned_owner_releases{0};
  auto orphaned_memory = make_owned_memory(orphaned_owner_releases);
  {
    auto buffer = grm::detail::DeviceBufferFactory::make_ready(
      device, orphaned_memory.pointer, 64, std::move(orphaned_memory.owner), failed_ops);
    buffer.reset();
  }
  assert(orphaned_owner_releases == 0);
}

void test_blocking_h2d_failure_safe_orphan()
{
  auto ops = std::make_shared<FakeOps>();
  ops->fail_h2d = true;
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto memory = make_owned_memory(owner_releases, 16);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 16, memory.owner, ops);
  const uint8_t source[4]{1, 2, 3, 4};
  expect_throws<std::runtime_error>(
    [&] { buffer->copy_from_host_blocking(source, sizeof(source)); });
  buffer.reset();
  memory.owner.reset();
  wait_for_cleanup();
  assert(owner_releases == 0);
}

void test_readiness_states_and_synchronized_writer()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 41, ops);
  std::atomic<int> owner_releases{0};
  auto memory = make_owned_memory(owner_releases, 32);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 32, memory.owner, ops);
  assert(buffer->readiness() == grm::BufferReadiness::kNotReady);
  {
    auto writer = buffer->get_write_handle(producer);
    writer.finalize();
  }
  assert(buffer->readiness() == grm::BufferReadiness::kEventBackedReady);
  buffer.reset();
  memory.owner.reset();
  wait_for_cleanup();
  assert(owner_releases == 1);

  std::atomic<int> sync_owner_releases{0};
  auto sync_memory = make_owned_memory(sync_owner_releases, 32);
  auto sync_buffer = grm::detail::DeviceBufferFactory::make_fresh(
    device, sync_memory.pointer, 32, sync_memory.owner, ops);
  {
    auto sync_writer = sync_buffer->get_synchronized_write_handle();
    sync_writer.finalize_synchronously();
  }
  assert(sync_buffer->readiness() == grm::BufferReadiness::kSynchronouslyReady);
  const int synchronizes_before_release = ops->synchronizes;
  {
    auto sync_lease = sync_buffer->get_blocking_ready_lease();
    static_cast<void>(sync_lease);
  }
  assert(ops->synchronizes == synchronizes_before_release);
  sync_buffer.reset();
  sync_memory.owner.reset();
  assert(sync_owner_releases == 1);
}

void test_producer_owner_is_retained_until_event_cleanup()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> allocation_releases{0};
  std::atomic<int> source_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 51, ops);
  auto memory = make_owned_memory(allocation_releases, 32);
  auto source_owner =
    std::shared_ptr<const void>(new int(7), [&source_releases](const void * value) {
      ++source_releases;
      delete static_cast<const int *>(value);
    });
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 32, memory.owner, ops);
  {
    auto writer = buffer->get_write_handle(producer);
    writer.retain_owner(source_owner);
    writer.finalize();
  }
  source_owner.reset();
  assert(source_releases == 0);
  buffer.reset();
  memory.owner.reset();
  wait_for_cleanup();
  assert(source_releases == 1);
  assert(allocation_releases == 1);
}

void test_synchronized_pool_cancel_and_failed_block_never_recycles()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
  {
    auto reservation = pool.acquire_synchronized_for(10ms);
    assert(reservation);
    reservation->writer.cancel();
  }
  assert(pool.available() == 1);

  {
    auto reservation = pool.acquire_synchronized_for(10ms);
    assert(reservation);
    reservation->writer.fail();
    assert(reservation->buffer->failed());
  }
  assert(pool.available() == 0);
  assert(!pool.shutdown(1ms));
}

void test_synchronized_pool_timeout_and_capacity_recovery()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto pool = grm::detail::PoolFactory::make(device, 32, 2, ops);
  auto first = pool.acquire_synchronized_for(10ms);
  auto second = pool.acquire_synchronized_for(10ms);
  assert(first && second);
  auto timed_out = pool.acquire_synchronized_for(1ms);
  assert(!timed_out);
  first->writer.cancel();
  second->writer.cancel();
  first.reset();
  second.reset();
  assert(pool.available() == 2);
  assert(pool.shutdown(10ms));
}

void test_failed_async_writer_never_recycles()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 61, ops);
  auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
  {
    auto block = std::make_unique<grm::PoolBlock>(pool.acquire(producer));
    block->writer.fail();
    assert(block->buffer->failed());
  }
  assert(pool.available() == 0);
  assert(!pool.shutdown(1ms));
}

void test_abandoned_async_writer_fails_and_never_recycles()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 63, ops);
  auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
  {
    auto block = std::make_unique<grm::PoolBlock>(pool.acquire(producer));
    // Deliberately omit finalize(), cancel(), and fail(). This models an
    // exception escaping a producer callback after work may have started.
  }
  assert(pool.available() == 0);
  assert(!pool.shutdown(1ms));
}

void test_async_writer_cancel_returns_reservation()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 62, ops);
  auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
  {
    auto block = std::make_unique<grm::PoolBlock>(pool.acquire(producer));
    block->writer.cancel();
  }
  assert(pool.available() == 1);
  assert(pool.shutdown(10ms));
}

void test_state_machine_and_multiple_readers()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  {
    const grm::DeviceId device{grm::BackendKind::kCuda, 2};
    ops->expected_event_device = device.ordinal;
    auto producer = make_stream(device, 11, ops);
    auto consumer_a = make_stream(device, 21, ops);
    auto consumer_b = make_stream(device, 22, ops);
    auto wrong_device = make_stream({grm::BackendKind::kCuda, 3}, 31, ops);
    auto memory = make_owned_memory(owner_releases);

    expect_throws<std::invalid_argument>(
      [&] { grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, {}, ops); });
    auto buffer =
      grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
    auto writer = buffer->get_write_handle(producer);
    assert(writer.data() == memory.pointer);
    expect_throws<std::logic_error>([&] { buffer->get_read_handle(consumer_a); });
    expect_throws<std::logic_error>([&] { buffer->get_write_handle(producer); });
    writer.finalize();
    writer.finalize();
    assert(ops->last_record_stream == 11);
    expect_throws<std::invalid_argument>([&] { buffer->get_read_handle(wrong_device); });

    auto reader_a = buffer->get_read_handle(consumer_a);
    auto reader_b = buffer->get_read_handle(consumer_b);
    assert(ops->waits == 2);
    assert(reader_a.data() == memory.pointer);
    reader_a.finish();
    reader_a.finish();
    reader_b.finish();
    assert(ops->records == 3);
    expect_throws<std::logic_error>([&] { buffer->get_write_handle(producer); });

    auto lease = buffer->get_blocking_ready_lease();
    assert(lease.data() == memory.pointer);
    assert(ops->synchronizes == 1);

    buffer.reset();
    memory.owner.reset();
    assert(owner_releases == 0);
  }
  wait_for_cleanup();
  assert(owner_releases == 1);
  assert(ops->event_device_failures == 0);
}

void test_handle_retains_allocation()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  {
    const grm::DeviceId device{grm::BackendKind::kCuda, 0};
    auto producer = make_stream(device, 1, ops);
    auto consumer = make_stream(device, 2, ops);
    auto memory = make_owned_memory(owner_releases, 8);
    auto buffer =
      grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 8, memory.owner, ops);
    {
      auto writer = buffer->get_write_handle(producer);
      writer.finalize();
    }
    auto retained_reader = buffer->get_read_handle(consumer);
    buffer.reset();
    memory.owner.reset();
    assert(owner_releases == 0);
    retained_reader.finish();
    assert(owner_releases == 0);
  }
  wait_for_cleanup();
  assert(owner_releases == 1);
}

void test_writer_event_failures_safe_orphan()
{
  for (const bool fail_during_record : {false, true}) {
    auto ops = std::make_shared<FakeOps>();
    std::atomic<int> owner_releases{0};
    const grm::DeviceId device{grm::BackendKind::kCuda, 0};
    auto producer = make_stream(device, 1, ops);
    auto memory = make_owned_memory(owner_releases);
    auto buffer =
      grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
    {
      auto writer = buffer->get_write_handle(producer);
      if (fail_during_record) {
        ops->fail_record = true;
      } else {
        ops->fail_create = true;
      }
      expect_throws<std::runtime_error>([&] { writer.finalize(); });
    }
    buffer.reset();
    memory.owner.reset();
    wait_for_cleanup();
    assert(owner_releases == 0);
    assert(ops->destroys == (fail_during_record ? 1 : 0));
    if (fail_during_record) {
      assert(ops->destroy_count(1) == 1);
    }
  }
}

void test_reader_event_failures_safe_orphan()
{
  for (const bool fail_during_record : {false, true}) {
    auto ops = std::make_shared<FakeOps>();
    std::atomic<int> owner_releases{0};
    const grm::DeviceId device{grm::BackendKind::kCuda, 0};
    auto producer = make_stream(device, 1, ops);
    auto consumer = make_stream(device, 2, ops);
    auto memory = make_owned_memory(owner_releases);
    auto buffer =
      grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
    buffer->get_write_handle(producer).finalize();
    {
      auto reader = buffer->get_read_handle(consumer);
      if (fail_during_record) {
        ops->fail_record = true;
      } else {
        ops->fail_create = true;
      }
      expect_throws<std::runtime_error>([&] { reader.finish(); });
    }
    buffer.reset();
    memory.owner.reset();
    wait_for_cleanup();
    assert(owner_releases == 0);
    assert(ops->destroy_count(1) == 1);
    if (fail_during_record) {
      assert(ops->destroy_count(2) == 1);
    }
  }
}

void test_wait_failure_allows_pool_recycle()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 1, ops);
  auto consumer = make_stream(device, 2, ops);
  auto pool = grm::detail::PoolFactory::make(device, 32, 1, ops);
  {
    auto held = std::make_unique<grm::PoolBlock>(pool.acquire(producer));
    held->writer.finalize();
    ops->fail_wait = true;
    expect_throws<std::runtime_error>([&] { held->buffer->get_read_handle(consumer); });
    assert(pool.available() == 0);
  }
  wait_for_cleanup();
  assert(pool.available() == 1);
  assert(ops->destroy_count(1) == 1);
  assert(ops->allocation_releases == 0);
}

void test_cleanup_failure_destroys_each_event_once_and_orphans()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 1, ops);
  auto consumer_a = make_stream(device, 2, ops);
  auto consumer_b = make_stream(device, 3, ops);
  auto memory = make_owned_memory(owner_releases);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
  buffer->get_write_handle(producer).finalize();
  buffer->get_read_handle(consumer_a).finish();
  buffer->get_read_handle(consumer_b).finish();
  ops->fail_synchronize(2);
  buffer.reset();
  memory.owner.reset();
  wait_for_cleanup();
  assert(owner_releases == 0);
  assert(ops->synchronizes == 3);
  assert(ops->destroy_count(1) == 1);
  assert(ops->destroy_count(2) == 1);
  assert(ops->destroy_count(3) == 1);
}

void test_blocking_failure_safe_orphan()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 1, ops);
  auto memory = make_owned_memory(owner_releases);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
  buffer->get_write_handle(producer).finalize();
  ops->fail_synchronize(1);
  expect_throws<std::runtime_error>([&] { buffer->get_blocking_ready_lease(); });
  buffer.reset();
  memory.owner.reset();
  wait_for_cleanup();
  assert(owner_releases == 0);
  assert(ops->destroy_count(1) == 1);
}

void test_pool_facade_can_be_destroyed_before_block()
{
  auto ops = std::make_shared<FakeOps>();
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 1, ops);
  auto pool = std::make_unique<grm::FixedDeviceMemoryPool>(
    grm::detail::PoolFactory::make(device, 32, 1, ops));
  auto held = std::make_unique<grm::PoolBlock>(pool->acquire(producer));
  pool.reset();
  held->writer.finalize();
  held.reset();
  wait_for_cleanup();
  assert(ops->allocation_releases == 1);
}

void test_pending_cleanup_drain()
{
  auto ops = std::make_shared<FakeOps>();
  ops->allow_synchronize = false;
  std::atomic<int> owner_releases{0};
  const grm::DeviceId device{grm::BackendKind::kCuda, 0};
  auto producer = make_stream(device, 1, ops);
  auto memory = make_owned_memory(owner_releases);
  auto buffer =
    grm::detail::DeviceBufferFactory::make_fresh(device, memory.pointer, 64, memory.owner, ops);
  buffer->get_write_handle(producer).finalize();
  buffer.reset();
  memory.owner.reset();

  assert(ops->wait_until_synchronize_entered(1s));
  assert(grm::detail::pending_release_count(grm::BackendKind::kCuda) == 1);
  assert(!grm::detail::wait_for_pending_releases(grm::BackendKind::kCuda, 1ms));
  assert(owner_releases == 0);
  ops->unblock_synchronize();
  wait_for_cleanup();
  assert(owner_releases == 1);
}

int main()
{
  static_assert(std::is_move_constructible_v<grm::WriteHandle>);
  static_assert(!std::is_move_assignable_v<grm::WriteHandle>);
  static_assert(std::is_move_constructible_v<grm::ReadHandle>);
  static_assert(!std::is_move_assignable_v<grm::ReadHandle>);
  static_assert(!std::is_move_assignable_v<grm::BlockingReadyLease>);
  static_assert(std::is_move_constructible_v<grm::SynchronizedWriteHandle>);
  static_assert(!std::is_move_assignable_v<grm::SynchronizedWriteHandle>);

  test_state_machine_and_multiple_readers();
  wait_for_cleanup();
  test_handle_retains_allocation();
  wait_for_cleanup();
  test_writer_event_failures_safe_orphan();
  test_reader_event_failures_safe_orphan();
  test_wait_failure_allows_pool_recycle();
  test_cleanup_failure_destroys_each_event_once_and_orphans();
  test_blocking_failure_safe_orphan();
  test_pool_facade_can_be_destroyed_before_block();
  test_pending_cleanup_drain();
  test_blocking_copy_api();
  test_synchronized_owner_releases_without_worker_thread();
  test_blocking_h2d_failure_safe_orphan();
  test_readiness_states_and_synchronized_writer();
  test_producer_owner_is_retained_until_event_cleanup();
  test_synchronized_pool_cancel_and_failed_block_never_recycles();
  test_synchronized_pool_timeout_and_capacity_recovery();
  test_failed_async_writer_never_recycles();
  test_abandoned_async_writer_fails_and_never_recycles();
  test_async_writer_cancel_returns_reservation();
  return 0;
}
