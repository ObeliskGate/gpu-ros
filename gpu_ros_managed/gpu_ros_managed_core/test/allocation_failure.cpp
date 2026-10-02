// Copyright 2026 gpu_ros_managed contributors
// Licensed under the Apache License, Version 2.0.
#include <cstdlib>
#include <new>

// Keep replaceable allocation functions in a separate translation unit so
// compiler allocation/deallocation analysis sees the matching new/delete API.
// Only the isolated pool-wrapper failure subprocess arms this hook.
bool should_fail_allocation(std::size_t size) noexcept;
void * operator new(std::size_t size)
{
  if (should_fail_allocation(size)) {
    throw std::bad_alloc();
  }
  if (void * allocation = std::malloc(size == 0 ? 1 : size)) {
    return allocation;
  }
  throw std::bad_alloc();
}

void * operator new[](std::size_t size)
{
  if (should_fail_allocation(size)) {
    throw std::bad_alloc();
  }
  if (void * allocation = std::malloc(size == 0 ? 1 : size)) {
    return allocation;
  }
  throw std::bad_alloc();
}

void operator delete(void * allocation) noexcept
{
  std::free(allocation);
}
void operator delete[](void * allocation) noexcept
{
  std::free(allocation);
}
void operator delete(void * allocation, std::size_t) noexcept
{
  std::free(allocation);
}
void operator delete[](void * allocation, std::size_t) noexcept
{
  std::free(allocation);
}
