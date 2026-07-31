#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
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
  grm::BackendKind kind() const noexcept override {return grm::BackendKind::kCuda;}

  void select_device(int ordinal) override
  {
    if (fail_select.exchange(false)) {throw std::runtime_error("select failure");}
    selected = ordinal;
  }

  grm::detail::Event create_event() override
  {
    if (fail_create.exchange(false)) {throw std::runtime_error("create failure");}
    std::lock_guard<std::mutex> lock(mutex);
    const auto event = ++next_event;
    live.insert(event);
    return event;
  }

  void record_event(
    grm::detail::Event event, grm::detail::NativeStream stream) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    require_live(event);
    last_record_stream = stream;
    ++records;
    if (fail_record.exchange(false)) {throw std::runtime_error("record failure");}
  }

  void wait_event(
    grm::detail::NativeStream stream, grm::detail::Event event) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    require_live(event);
    last_wait_stream = stream;
    ++waits;
    if (fail_wait.exchange(false)) {throw std::runtime_error("wait failure");}
  }

  void synchronize_event(grm::detail::Event event) override
  {
    std::unique_lock<std::mutex> lock(mutex);
    require_live(event);
    ++synchronizes;
    synchronize_entered = true;
    synchronize_entered_cv.notify_all();
    synchronize_gate_cv.wait(lock, [&] {return allow_synchronize;});
    if (fail_synchronize_events.erase(event) != 0) {
      throw std::runtime_error("synchronize failure");
    }
  }

  void destroy_event(grm::detail::Event event) noexcept override
  {
    std::lock_guard<std::mutex> lock(mutex);
    live.erase(event);
    ++destroy_counts[event];
    ++destroys;
  }

  std::shared_ptr<void> allocate_device(int ordinal, size_t bytes) override
  {
    selected = ordinal;
    void * pointer = std::malloc(bytes);
    if (pointer == nullptr) {throw std::bad_alloc();}
    return std::shared_ptr<void>(pointer, [this](void * value) {
      ++allocation_releases;
      std::free(value);
    });
  }

  void copy_device_to_host(int, void *, const void *, size_t) override {}

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
    return synchronize_entered_cv.wait_for(lock, timeout, [&] {return synchronize_entered;});
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
  std::atomic<int> selected{-1};
  grm::detail::NativeStream last_record_stream{0};
  grm::detail::NativeStream last_wait_stream{0};
  bool allow_synchronize{true};

private:
  void require_live(grm::detail::Event event)
  {
    if (live.count(event) == 0) {throw std::runtime_error("unknown event");}
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

template<typename Exception, typename Function>
void expect_throws(Function && function)
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
  if (pointer == nullptr) {throw std::bad_alloc();}
  return {
    pointer,
    std::shared_ptr<void>(pointer, [&releases](void * value) {
      ++releases;
      std::free(value);
    })
  };
}

grm::DeviceStream make_stream(
  const grm::DeviceId & device, grm::detail::NativeStream native,
  const std::shared_ptr<FakeOps> & ops)
{
  return grm::detail::DeviceBufferFactory::make_stream(device, native, {}, ops);
}

void wait_for_cleanup()
{
  assert(grm::detail::wait_for_pending_releases(grm::BackendKind::kCuda, 2s));
  assert(grm::detail::pending_release_count(grm::BackendKind::kCuda) == 0);
}

void test_state_machine_and_multiple_readers()
{
  auto ops = std::make_shared<FakeOps>();
  std::atomic<int> owner_releases{0};
  {
    const grm::DeviceId device{grm::BackendKind::kCuda, 2};
    auto producer = make_stream(device, 11, ops);
    auto consumer_a = make_stream(device, 21, ops);
    auto consumer_b = make_stream(device, 22, ops);
    auto wrong_device = make_stream({grm::BackendKind::kCuda, 3}, 31, ops);
    auto memory = make_owned_memory(owner_releases);

    expect_throws<std::invalid_argument>([&] {
      grm::detail::DeviceBufferFactory::make_fresh(
        device, memory.pointer, 64, {}, ops);
    });
    auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
      device, memory.pointer, 64, memory.owner, ops);
    auto writer = buffer->get_write_handle(producer);
    assert(writer.data() == memory.pointer);
    expect_throws<std::logic_error>([&] {buffer->get_read_handle(consumer_a);});
    expect_throws<std::logic_error>([&] {buffer->get_write_handle(producer);});
    writer.finalize();
    writer.finalize();
    assert(ops->last_record_stream == 11);
    expect_throws<std::invalid_argument>([&] {buffer->get_read_handle(wrong_device);});

    auto reader_a = buffer->get_read_handle(consumer_a);
    auto reader_b = buffer->get_read_handle(consumer_b);
    assert(ops->waits == 2);
    assert(reader_a.data() == memory.pointer);
    reader_a.finish();
    reader_a.finish();
    reader_b.finish();
    assert(ops->records == 3);
    expect_throws<std::logic_error>([&] {buffer->get_write_handle(producer);});

    auto lease = buffer->get_blocking_ready_lease();
    assert(lease.data() == memory.pointer);
    assert(ops->synchronizes == 1);

    buffer.reset();
    memory.owner.reset();
    assert(owner_releases == 0);
  }
  wait_for_cleanup();
  assert(owner_releases == 1);
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
    auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
      device, memory.pointer, 8, memory.owner, ops);
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
    auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
      device, memory.pointer, 64, memory.owner, ops);
    {
      auto writer = buffer->get_write_handle(producer);
      if (fail_during_record) {
        ops->fail_record = true;
      } else {
        ops->fail_create = true;
      }
      expect_throws<std::runtime_error>([&] {writer.finalize();});
    }
    buffer.reset();
    memory.owner.reset();
    wait_for_cleanup();
    assert(owner_releases == 0);
    assert(ops->destroys == (fail_during_record ? 1 : 0));
    if (fail_during_record) {assert(ops->destroy_count(1) == 1);}
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
    auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
      device, memory.pointer, 64, memory.owner, ops);
    buffer->get_write_handle(producer).finalize();
    {
      auto reader = buffer->get_read_handle(consumer);
      if (fail_during_record) {
        ops->fail_record = true;
      } else {
        ops->fail_create = true;
      }
      expect_throws<std::runtime_error>([&] {reader.finish();});
    }
    buffer.reset();
    memory.owner.reset();
    wait_for_cleanup();
    assert(owner_releases == 0);
    assert(ops->destroy_count(1) == 1);
    if (fail_during_record) {assert(ops->destroy_count(2) == 1);}
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
    expect_throws<std::runtime_error>([&] {held->buffer->get_read_handle(consumer);});
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
  auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
    device, memory.pointer, 64, memory.owner, ops);
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
  auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
    device, memory.pointer, 64, memory.owner, ops);
  buffer->get_write_handle(producer).finalize();
  ops->fail_synchronize(1);
  expect_throws<std::runtime_error>([&] {buffer->get_blocking_ready_lease();});
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
  auto buffer = grm::detail::DeviceBufferFactory::make_fresh(
    device, memory.pointer, 64, memory.owner, ops);
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
  return 0;
}
