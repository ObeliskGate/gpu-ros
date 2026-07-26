# syntax=docker/dockerfile:1

ARG AMD_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.1.1-complete

# ORT 1.23.1 is the version used by the Phase 1 NVIDIA baseline. AMD's
# supported pairing for that release is ROCm 7.1/7.1.1.
FROM ${AMD_BASE_IMAGE} AS rocm-migraphx

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    half \
    migraphx \
    migraphx-dev \
    && rm -rf /var/lib/apt/lists/*

FROM rocm-migraphx AS onnxruntime-builder

ARG ORT_VERSION=1.23.1
ARG ORT_BUILD_JOBS=16
ARG AMD_GPU_TARGETS=gfx942

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    patch \
    python3 \
    python3-dev \
    python3-packaging \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src
RUN git clone --branch "v${ORT_VERSION}" --depth 1 \
      --recursive --shallow-submodules \
      https://github.com/microsoft/onnxruntime.git

WORKDIR /opt/src/onnxruntime
# Backport the Linux MIGraphX test-build fix merged upstream after ORT 1.23.x.
# Keep unit-test targets and warnings-as-errors enabled.
COPY docker/patches/onnxruntime-1.23.1-migraphx-linux-unused-helper.patch /tmp/
RUN patch -p1 < /tmp/onnxruntime-1.23.1-migraphx-linux-unused-helper.patch

# MIGraphX 7.1.1 implements GridSample, but ORT 1.23.1 omits it from the
# MIGraphX EP capability allowlist. Fail if the pinned source no longer matches.
COPY docker/patches/onnxruntime-1.23.1-migraphx-enable-gridsample.patch /tmp/
RUN git apply --check /tmp/onnxruntime-1.23.1-migraphx-enable-gridsample.patch \
    && git apply /tmp/onnxruntime-1.23.1-migraphx-enable-gridsample.patch

RUN ./build.sh \
      --config Release \
      --parallel "${ORT_BUILD_JOBS}" \
      --build_shared_lib \
      --skip_tests \
      --allow_running_as_root \
      --use_migraphx \
      --migraphx_home /opt/rocm \
      --cmake_extra_defines \
        GPU_TARGETS="${AMD_GPU_TARGETS}" \
        CMAKE_HIP_ARCHITECTURES="${AMD_GPU_TARGETS}"

# ORT does not publish a standalone C++ MIGraphX archive. Assemble the same
# include/lib layout consumed by the ROS package from the source build.
RUN mkdir -p /opt/onnxruntime/include /opt/onnxruntime/lib \
    && cp -a include/onnxruntime/core/session/*.h /opt/onnxruntime/include/ \
    && find build/Linux/Release -maxdepth 1 \
      \( -type f -o -type l \) -name 'libonnxruntime*.so*' \
      -exec cp -a {} /opt/onnxruntime/lib/ \; \
    && test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h \
    && test -e /opt/onnxruntime/lib/libonnxruntime.so \
    && test -e /opt/onnxruntime/lib/libonnxruntime_providers_migraphx.so

FROM rocm-migraphx AS ros-runtime-base

ARG ROS_DISTRO=jazzy
ARG DEBIAN_FRONTEND=noninteractive
ARG ORT_VERSION=1.23.1

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ORT_VERSION=${ORT_VERSION}
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

COPY --from=onnxruntime-builder /opt/onnxruntime /opt/onnxruntime
RUN echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig \
    && test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h \
    && test -e /opt/onnxruntime/lib/libonnxruntime.so

# ROS 2 apt repository. Keep this layer independent of project source edits.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libopencv-dev \
    pkg-config \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-onnx \
    python3-pip \
    python3-psutil \
    python3-rosdep \
    python3-vcstool \
    python3-yaml \
    ros-${ROS_DISTRO}-ament-cmake-auto \
    ros-${ROS_DISTRO}-ament-cmake-gtest \
    ros-${ROS_DISTRO}-ament-cmake-python \
    ros-${ROS_DISTRO}-ament-lint-auto \
    ros-${ROS_DISTRO}-ament-lint-common \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-launch-testing-ament-cmake \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-rclcpp-action \
    ros-${ROS_DISTRO}-rclcpp-components \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-rosbag2-compression-zstd \
    ros-${ROS_DISTRO}-rosbag2-cpp \
    ros-${ROS_DISTRO}-rosbag2-py \
    ros-${ROS_DISTRO}-rosbag2-storage \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-default-runtime \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true
RUN rosdep update --rosdistro ${ROS_DISTRO}

FROM ros-runtime-base AS ros2-benchmark-builder

ARG ROS2_BENCHMARK_REF=v4.5-0

WORKDIR /opt/src
RUN git clone --branch "${ROS2_BENCHMARK_REF}" --depth 1 \
      https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark.git

COPY docker/patches/ros2-benchmark-v4.5-standalone.patch /tmp/
WORKDIR /opt/src/ros2_benchmark
RUN git apply --check /tmp/ros2-benchmark-v4.5-standalone.patch \
    && git apply /tmp/ros2-benchmark-v4.5-standalone.patch

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon --log-base /tmp/ros2_benchmark_log build \
      --merge-install \
      --install-base /opt/ros2_benchmark \
      --build-base /tmp/ros2_benchmark_build \
      --base-paths \
        /opt/src/ros2_benchmark/ros2_benchmark_interfaces \
        /opt/src/ros2_benchmark/ros2_benchmark \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && source /opt/ros2_benchmark/setup.bash \
    && ros2 pkg prefix ros2_benchmark \
    && ros2 pkg prefix ros2_benchmark_interfaces

FROM ros-runtime-base AS runtime

COPY --from=ros2-benchmark-builder /opt/ros2_benchmark /opt/ros2_benchmark

COPY docker/phase2a-amd-entrypoint.sh /usr/local/bin/phase2a-amd-entrypoint.sh
RUN chmod +x /usr/local/bin/phase2a-amd-entrypoint.sh

WORKDIR /workspaces/amd_ros_object_detection
ENTRYPOINT ["/usr/local/bin/phase2a-amd-entrypoint.sh"]
CMD ["bash"]
