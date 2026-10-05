# Required official Isaac ROS 5.0 image reference with a verified digest; there is no default.
ARG ISAAC_ROS_BASE_IMAGE
FROM ${ISAAC_ROS_BASE_IMAGE}

ARG ISAAC_ROS_BASE_IMAGE
ENV ROS_DISTRO=lyrical
ENV GPU_ROS_NVIDIA_PROFILE=1
LABEL com.gpu-ros.isaac-ros-base-image="${ISAAC_ROS_BASE_IMAGE}"
RUN bash -c '[[ "${ISAAC_ROS_BASE_IMAGE}" =~ ^nvcr\.io/nvidia/isaac/ros(:[^@[:space:]]+)?@sha256:[0-9a-f]{64}$ ]] || { echo "BLOCKED: ISAAC_ROS_BASE_IMAGE must be the verified official Isaac ROS 5.0 image reference with sha256 digest." >&2; exit 1; }'

# ONNX Runtime — reuse the CUDA-13-matched build that ships with Triton
# (/opt/tritonserver/backends/onnxruntime, ORT 1.23.1). We only fetch matching
# headers (header API is CUDA-version-independent); the .so comes from Triton.
COPY gpu_ros_object_detection/config/onnxruntime.lock /tmp/onnxruntime.lock
RUN . /tmp/onnxruntime.lock && wget -q -O /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 --wildcards "*/include/*" && \
    rm /tmp/ort.tgz && \
    echo "/opt/tritonserver/backends/onnxruntime" > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig

RUN bash -lc '[[ -f /opt/ros/lyrical/setup.bash ]] || { echo "BLOCKED: official image does not provide ROS Lyrical." >&2; exit 1; } \
  && source /opt/ros/lyrical/setup.bash \
  && test "${ROS_DISTRO}" = lyrical \
  && for package in rosidl_buffer cuda_buffer cuda_buffer_backend \
    isaac_ros_tensor_list_interfaces isaac_ros_benchmark ros2_benchmark \
    isaac_ros_image_proc isaac_ros_tensor_proc isaac_ros_rtdetr \
    isaac_ros_yolov8 isaac_ros_tensor_rt; do \
      ros2 pkg prefix "${package}" || { echo "BLOCKED: ${package} is missing from the official image; a paired source overlay is required." >&2; exit 1; }; \
    done'
# Python dependencies for model preparation (FP16 conversion).
RUN pip install --no-cache-dir --break-system-packages onnxconverter-common
