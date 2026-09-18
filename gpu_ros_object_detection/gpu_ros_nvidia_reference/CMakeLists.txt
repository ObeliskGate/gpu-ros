cmake_minimum_required(VERSION 3.22.1)
project(gpu_ros_nvidia_reference LANGUAGES NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
