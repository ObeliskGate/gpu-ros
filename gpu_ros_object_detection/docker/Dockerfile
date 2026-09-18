FROM nvcr.io/nvidia/isaac/ros:isaac_ros_89df02a734965ed64c227ef531c09d65-amd64

# 1. Benchmark packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-isaac-ros-rtdetr-benchmark \
    ros-jazzy-isaac-ros-detectnet-benchmark \
    ros-jazzy-isaac-ros-grounding-dino-benchmark \
    && rm -rf /var/lib/apt/lists/*

# 2. Isaac ROS test framework. Keep recommended dependencies enabled because
# python3-torch-pip-shim needs them while its package scripts install PyTorch.
RUN apt-get update && apt-get install -y \
    ros-jazzy-isaac-ros-test \
    && rm -rf /var/lib/apt/lists/*

# 3. ONNX Runtime — reuse the CUDA-13-matched build that ships with Triton
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

# 4. Python deps for model preparation (FP16 conversion).
RUN pip install --no-cache-dir --break-system-packages onnxconverter-common
