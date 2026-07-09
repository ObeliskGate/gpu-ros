# syntax=docker/dockerfile:1

ARG AMD_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.2-complete
FROM ${AMD_BASE_IMAGE}

ARG ROS_DISTRO=jazzy
ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_BENCHMARK_DEPS=0

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV ONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include
ENV ONNXRUNTIME_LIBRARY=/opt/onnxruntime/lib/libonnxruntime.so
ENV COLCON_DEFAULTS_FILE=/workspaces/amd_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    locales \
    lsb-release \
    software-properties-common \
    sudo \
    wget \
    && locale-gen en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 apt repository. Keep this layer independent of project source edits.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    pkg-config \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    ros-${ROS_DISTRO}-ament-cmake-auto \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-launch-testing-ament-cmake \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-rclcpp-components \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Message-only Isaac ROS dependency used by Phase 2a. Install it when the apt
# repo is configured; otherwise leave a clear build-time note and require the
# package to be present in the colcon workspace.
RUN apt-get update \
    && if apt-cache show ros-${ROS_DISTRO}-isaac-ros-tensor-list-interfaces >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-isaac-ros-tensor-list-interfaces; \
    else \
      echo "ros-${ROS_DISTRO}-isaac-ros-tensor-list-interfaces not found in apt; provide the official package in the workspace or configure the Isaac ROS apt source."; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Optional benchmark harness dependencies. Leave disabled for a lean build
# image; enable with --build-arg INSTALL_BENCHMARK_DEPS=1 when the required apt
# packages are available for the target environment.
RUN if [[ "${INSTALL_BENCHMARK_DEPS}" == "1" ]]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros2-benchmark \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN rosdep init 2>/dev/null || true
RUN rosdep update --rosdistro ${ROS_DISTRO}

COPY docker/phase2a-amd-entrypoint.sh /usr/local/bin/phase2a-amd-entrypoint.sh
RUN chmod +x /usr/local/bin/phase2a-amd-entrypoint.sh

WORKDIR /workspaces/amd_ros_object_detection
ENTRYPOINT ["/usr/local/bin/phase2a-amd-entrypoint.sh"]
CMD ["bash"]
