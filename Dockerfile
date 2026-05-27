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
    && rm -rf /var/lib/apt/lists/*
# NOTE: ONNX Runtime 1.20.1 is pre-installed in the Isaac ROS base image at /opt/onnxruntime.
