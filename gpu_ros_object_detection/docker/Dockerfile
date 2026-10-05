# Required official Isaac ROS 5.0 image reference with a verified digest; there is no default.
ARG ISAAC_ROS_BASE_IMAGE
FROM ${ISAAC_ROS_BASE_IMAGE}

ARG ISAAC_ROS_BASE_IMAGE
ENV ROS_DISTRO=lyrical
ENV GPU_ROS_NVIDIA_PROFILE=1
LABEL com.gpu-ros.isaac-ros-base-image="${ISAAC_ROS_BASE_IMAGE}"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY gpu_ros_object_detection/config/nvidia-runtime.lock /opt/gpu-ros/nvidia-runtime.lock
RUN source /opt/gpu-ros/nvidia-runtime.lock && \
    [[ "${ISAAC_ROS_BASE_IMAGE}" =~ ^nvcr\.io/nvidia/isaac/ros(:[^@[:space:]]+)?@sha256:${ISAAC_ROS_BASE_DIGEST}$ ]] || \
    { echo "ISAAC_ROS_BASE_IMAGE must name the locked official Isaac ROS 5.0 digest." >&2; exit 1; }

# The official base is not the previously derived benchmark image. Resolve its
# missing graph/CUDA/TensorRT dependencies at the recorded versions, never by
# upgrading an existing package. Its signed Isaac/CUDA/Ubuntu APT sources remain
# authoritative. The installer checks both the plan and downloaded archive hashes.
COPY gpu_ros_object_detection/config/nvidia-runtime-packages.lock /opt/gpu-ros/nvidia-runtime-packages.lock
COPY gpu_ros_object_detection/tools/install-nvidia-runtime-packages.py /opt/gpu-ros/install-nvidia-runtime-packages.py
RUN apt-get update && \
    python3 /opt/gpu-ros/install-nvidia-runtime-packages.py /opt/gpu-ros/nvidia-runtime-packages.lock && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

# Install the complete matching CUDA 13 ORT distribution, including its licenses.
# There is no Triton library/header split and no TensorRT EP selection.
COPY gpu_ros_object_detection/config/onnxruntime.lock /opt/gpu-ros/onnxruntime.lock
RUN source /opt/gpu-ros/onnxruntime.lock && \
    curl --fail --location --silent --show-error \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz" \
      --output /tmp/ort.tgz && \
    echo "${ORT_CUDA13_ARCHIVE_SHA256}  /tmp/ort.tgz" | sha256sum --check --strict && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    test -f /opt/onnxruntime/include/onnxruntime_cxx_api.h && \
    test -f "/opt/onnxruntime/lib/libonnxruntime.so.${ORT_VERSION}" && \
    test -f /opt/onnxruntime/LICENSE && \
    rm /tmp/ort.tgz

# Only the released plugin is built. SDK cuda_buffer, messages, registry and ROS
# core remain binary underlays; no rosdep transaction or sibling-package overlay.
RUN source /opt/gpu-ros/nvidia-runtime.lock && \
    test "$(dpkg-query -W -f='${Version}' ros-lyrical-cuda-buffer)" = "${CUDA_BUFFER_DEB_VERSION}" && \
    curl --fail --location --silent --show-error \
      "https://codeload.github.com/ros2/rosidl_buffer_backends/tar.gz/${CUDA_BUFFER_BACKEND_COMMIT}" \
      --output /tmp/cuda-buffer-backend.tgz && \
    echo "${CUDA_BUFFER_BACKEND_ARCHIVE_SHA256}  /tmp/cuda-buffer-backend.tgz" | sha256sum --check --strict && \
    mkdir -p /tmp/cuda-buffer-backend-source && \
    tar -xzf /tmp/cuda-buffer-backend.tgz -C /tmp/cuda-buffer-backend-source --strip-components=1 && \
    source /opt/ros/lyrical/setup.bash && \
    colcon --log-base /tmp/cuda-buffer-backend-log build \
      --base-paths /tmp/cuda-buffer-backend-source/cuda_buffer_backend/cuda_buffer_backend \
      --packages-select cuda_buffer_backend --merge-install \
      --build-base /tmp/cuda-buffer-backend-build \
      --install-base /opt/gpu-ros/cuda-buffer-backend \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DBUILD_TESTING=OFF \
      -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 && \
    install -m 0644 /tmp/cuda-buffer-backend-source/LICENSE \
      /opt/gpu-ros/cuda-buffer-backend/share/cuda_buffer_backend/LICENSE && \
    rm -rf /tmp/cuda-buffer-backend-source /tmp/cuda-buffer-backend-build \
      /tmp/cuda-buffer-backend-log /tmp/cuda-buffer-backend.tgz

# ENV covers direct docker exec/Compose commands as well as shells; the helper
# provides ordered activation after callers explicitly source the ROS underlay.
# Do not replace an Isaac workspace ENTRYPOINT supplied by the runtime launcher.
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV AMENT_PREFIX_PATH=/opt/gpu-ros/cuda-buffer-backend:/opt/ros/lyrical
ENV CMAKE_PREFIX_PATH=/opt/gpu-ros/cuda-buffer-backend:/opt/ros/lyrical
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/opt/gpu-ros/cuda-buffer-backend/lib:/opt/ros/lyrical/lib:/usr/local/cuda-13.2/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY gpu_ros_object_detection/tools/nvidia-runtime-setup.bash /opt/gpu-ros/setup.bash
RUN printf '%s\n' /opt/onnxruntime/lib /opt/gpu-ros/cuda-buffer-backend/lib \
      > /etc/ld.so.conf.d/gpu-ros-nvidia.conf && ldconfig && \
    source /opt/gpu-ros/setup.bash && \
    for package in rosidl_buffer cuda_buffer cuda_buffer_backend \
      isaac_ros_tensor_msgs tensor_msgs isaac_ros_benchmark ros2_benchmark \
      isaac_ros_dnn_image_encoder isaac_ros_image_proc isaac_ros_tensor_proc \
      isaac_ros_rtdetr isaac_ros_yolov8 isaac_ros_tensor_rt; do \
      ros2 pkg prefix "${package}" || exit 1; \
    done && \
    test "$(ros2 pkg prefix cuda_buffer_backend)" = /opt/gpu-ros/cuda-buffer-backend && \
    test "$(ros2 pkg prefix cuda_buffer)" = /opt/ros/lyrical

# Python dependencies for model preparation (FP16 conversion).
RUN source /opt/gpu-ros/nvidia-runtime.lock && \
    curl --fail --location --silent --show-error \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      --output /tmp/uv.tgz && \
    echo "${UV_LINUX_ARCHIVE_SHA256}  /tmp/uv.tgz" | sha256sum --check --strict && \
    tar -xzf /tmp/uv.tgz -C /tmp && \
    install /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv && \
    mkdir -p /usr/local/share/licenses/uv && \
    curl --fail --location --silent --show-error \
      "https://raw.githubusercontent.com/astral-sh/uv/${UV_VERSION}/LICENSE-MIT" \
      --output /usr/local/share/licenses/uv/LICENSE-MIT && \
    curl --fail --location --silent --show-error \
      "https://raw.githubusercontent.com/astral-sh/uv/${UV_VERSION}/LICENSE-APACHE" \
      --output /usr/local/share/licenses/uv/LICENSE-APACHE && \
    uv pip install --system --no-cache --break-system-packages \
      "onnxconverter-common==${ONNXCONVERTER_COMMON_VERSION}" && \
    rm -rf /tmp/uv-x86_64-unknown-linux-gnu /tmp/uv.tgz
