cmake_minimum_required(VERSION 3.22)
project(gpu_ros_managed_core_standalone_consumer LANGUAGES CXX)

find_package(gpu_ros_managed_core CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE gpu_ros_managed::core)
