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

ARG ORT_BUILD_JOBS=16
ARG AMD_GPU_TARGETS

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgmock-dev \
    libgtest-dev \
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
COPY gpu_ros_object_detection/config/onnxruntime.lock /tmp/onnxruntime.lock
RUN . /tmp/onnxruntime.lock \
    && git clone --branch "v${ORT_VERSION}" --depth 1 \
      --recursive --shallow-submodules \
      https://github.com/microsoft/onnxruntime.git \
    && test "$(git -C onnxruntime rev-parse HEAD)" = "${ORT_COMMIT}"

WORKDIR /opt/src/onnxruntime
# Backport the Linux MIGraphX test-build fix merged upstream after ORT 1.23.x.
# Keep unit-test targets and warnings-as-errors enabled.
COPY gpu_ros_object_detection/docker/patches/onnxruntime-1.23.1-migraphx-linux-unused-helper.patch /tmp/
RUN patch -p1 < /tmp/onnxruntime-1.23.1-migraphx-linux-unused-helper.patch

# MIGraphX 7.1.1 implements GridSample, but ORT 1.23.1 omits it from the
# MIGraphX EP capability allowlist. Fail if the pinned source no longer matches.
COPY gpu_ros_object_detection/docker/patches/onnxruntime-1.23.1-migraphx-enable-gridsample.patch /tmp/
RUN git apply --check /tmp/onnxruntime-1.23.1-migraphx-enable-gridsample.patch \
    && git apply /tmp/onnxruntime-1.23.1-migraphx-enable-gridsample.patch

# MIGraphX 2.14 can lose int64 division truncation while converting unsupported
# GPU pointwise types through float. Keep only int64 Div on CPU so RT-DETR box
# selection remains correct while the surrounding graph stays on MIGraphX.
COPY gpu_ros_object_detection/docker/patches/onnxruntime-1.23.1-migraphx-int64-div-cpu-fallback.patch /tmp/
RUN git apply --check /tmp/onnxruntime-1.23.1-migraphx-int64-div-cpu-fallback.patch \
    && git apply /tmp/onnxruntime-1.23.1-migraphx-int64-div-cpu-fallback.patch

RUN CMAKE_TARGETS="$(printf '%s' "${AMD_GPU_TARGETS}" | tr ',' ';')" \
    && ./build.sh \
      --config Release \
      --parallel "${ORT_BUILD_JOBS}" \
      --build_shared_lib \
      --skip_tests \
      --allow_running_as_root \
      --use_migraphx \
      --migraphx_home /opt/rocm \
      --cmake_extra_defines \
        GPU_TARGETS="${CMAKE_TARGETS}" \
        CMAKE_HIP_ARCHITECTURES="${CMAKE_TARGETS}"

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

ARG ROS_DISTRO=lyrical
ARG DEBIAN_FRONTEND=noninteractive
ARG ISAAC_ROS_BASE_IMAGE
ARG ROS_LYRICAL_APT_PACKAGE_SPECS

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=${ROS_DISTRO}
ENV GPU_ROS_NVIDIA_PROFILE=0
ENV OVG_WORKSPACE_ROOT=/workspaces/gpu-ros
ENV GPU_ROS_REPO_ROOT=/workspaces/gpu-ros
ENV OVG_ORT_STATE_ROOT=/workspaces/ovg-ort
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV ONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include
ENV ONNXRUNTIME_LIBRARY=/opt/onnxruntime/lib/libonnxruntime.so
ENV COLCON_DEFAULTS_FILE=/workspaces/gpu-ros/gpu_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    locales \
    lsb-release \
    software-properties-common \
    sudo \
    unzip \
    python3 \
    wget \
    && locale-gen en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime-builder /opt/onnxruntime /opt/onnxruntime
RUN echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig \
    && test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h \
    && test -e /opt/onnxruntime/lib/libonnxruntime.so

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libopencv-dev \
    pkg-config \
    libgmock-dev \
    libgtest-dev \
    ninja-build \
    patch \
    python3-dev \
    python3-numpy \
    python3-onnx \
    python3-packaging \
    python3-pip \
    python3-psutil \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

COPY gpu_ros_object_detection/docker/install_isaac_ros_apt_packages.sh /usr/local/bin/install-isaac-ros-apt-packages
RUN bash /usr/local/bin/install-isaac-ros-apt-packages



FROM ros-runtime-base AS ros2-benchmark-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG ROS2_BENCHMARK_COMMIT

WORKDIR /opt/src
RUN [[ "${ROS2_BENCHMARK_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || { \
      echo "BLOCKED: ROS2_BENCHMARK_COMMIT must be the exact image-paired source SHA." >&2; \
      exit 1; \
    } \
    && git init ros2_benchmark \
    && git -C ros2_benchmark remote add origin https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark.git \
    && git -C ros2_benchmark fetch --depth=1 origin "${ROS2_BENCHMARK_COMMIT}" \
    && git -C ros2_benchmark checkout --detach FETCH_HEAD \
    && test "$(git -C ros2_benchmark rev-parse HEAD)" = "${ROS2_BENCHMARK_COMMIT}"

COPY gpu_ros_object_detection/docker/prepare_ros2_benchmark_standalone.py /usr/local/bin/prepare-ros2-benchmark-standalone
RUN python3 /usr/local/bin/prepare-ros2-benchmark-standalone \
      --source-root /opt/src/ros2_benchmark \
      --expected-commit "${ROS2_BENCHMARK_COMMIT}" \
    && git -C /opt/src/ros2_benchmark diff --binary \
      > /tmp/ros2-benchmark-standalone.patch \
    && test -s /tmp/ros2-benchmark-standalone.patch \
    && git -C /opt/src/ros2_benchmark checkout -- \
      ros2_benchmark/CMakeLists.txt \
      ros2_benchmark/package.xml \
      ros2_benchmark_interfaces/CMakeLists.txt \
      ros2_benchmark_interfaces/package.xml \
    && git -C /opt/src/ros2_benchmark apply --check \
      /tmp/ros2-benchmark-standalone.patch \
    && git -C /opt/src/ros2_benchmark apply \
      /tmp/ros2-benchmark-standalone.patch \
    && git -C /opt/src/ros2_benchmark diff --check

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

# The runtime is also the external ONNX Runtime development environment.  Keep
# The build toolchain needed by gpu_ros_object_detection/tools/build-phase2a-external-ort.sh
# in the final image so that source, patches, build trees and installs can remain on
# the host/SIF bind mounts rather than being hidden in an image layer.

ARG AMD_BASE_IMAGE
ARG AMD_GPU_TARGETS
ARG ROS_DISTRO=lyrical
ARG ROS2_BENCHMARK_COMMIT
ARG ISAAC_ROS_BASE_IMAGE
ARG ROS_LYRICAL_APT_PACKAGE_SPECS
COPY gpu_ros_object_detection/config/onnxruntime.lock /tmp/onnxruntime.lock
RUN . /tmp/onnxruntime.lock \
    && ROS_PACKAGE_LOCK_SHA256="$(python3 -c 'import json; print(json.load(open("/var/lib/gpu-ros/isaac-ros-5-ros-package-lock.json"))["package_specs_sha256"])')" \
    && mkdir -p /opt/ovg \
    && printf '{"base_image":"%s","rocm":"7.1.1","ort":"%s","ros_distro":"%s","isaac_ros_base_image":"%s","ros_package_lock_sha256":"%s","ros2_benchmark_commit":"%s","gpu_targets":"%s","provider_patches":["migraphx-enable-gridsample","migraphx-int64-div-cpu-fallback-v1"]}\n' \
      "${AMD_BASE_IMAGE:-rocm/dev-ubuntu-24.04:7.1.1-complete}" \
      "${ORT_VERSION}" "${ROS_DISTRO}" "${ISAAC_ROS_BASE_IMAGE}" \
      "${ROS_PACKAGE_LOCK_SHA256}" "${ROS2_BENCHMARK_COMMIT}" "${AMD_GPU_TARGETS:-}" \
      > /opt/ovg/image-manifest.json

COPY gpu_ros_object_detection/docker/phase2a-amd-entrypoint.sh /usr/local/bin/phase2-amd-entrypoint.sh
RUN chmod +x /usr/local/bin/phase2-amd-entrypoint.sh \
    && ln -s phase2-amd-entrypoint.sh /usr/local/bin/phase2a-amd-entrypoint.sh

WORKDIR /workspaces/gpu-ros
ENTRYPOINT ["/usr/local/bin/phase2-amd-entrypoint.sh"]
CMD ["bash"]
