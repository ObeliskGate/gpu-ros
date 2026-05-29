FROM nvcr.io/nvidia/isaac/ros:isaac_ros_28556f8bc78a98822bd08b2d7c6fcf9b-amd64

# 1. Tooling needed to fetch the NGC CLI (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. NGC CLI (only changes when bumping NGC_CLI_VERSION)
ARG NGC_CLI_VERSION=4.18.0
RUN wget -q -O /tmp/ngccli.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/${NGC_CLI_VERSION}/files/ngccli_linux.zip && \
    unzip -q /tmp/ngccli.zip -d /opt && rm /tmp/ngccli.zip
ENV PATH="/opt/ngc-cli:${PATH}"

# 3. Benchmark packages (this is the layer you'll iterate on; keep it last)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-isaac-ros-rtdetr-benchmark \
    ros-jazzy-isaac-ros-detectnet-benchmark \
    ros-jazzy-isaac-ros-grounding-dino-benchmark \
    ros-jazzy-isaac-ros-test \
    && rm -rf /var/lib/apt/lists/*
# 4. ONNX Runtime — reuse the CUDA-13-matched build that ships with Triton
# (/opt/tritonserver/backends/onnxruntime, ORT 1.23.1). We only fetch matching
# headers (header API is CUDA-version-independent); the .so comes from Triton.
ARG ORT_VERSION=1.23.1
RUN wget -q -O /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 --wildcards "*/include/*" && \
    rm /tmp/ort.tgz && \
    echo "/opt/tritonserver/backends/onnxruntime" > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig
