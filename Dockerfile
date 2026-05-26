FROM nvcr.io/nvidia/isaac/ros:isaac_ros_28556f8bc78a98822bd08b2d7c6fcf9b-amd64

# Benchmark dependencies (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-isaac-ros-rtdetr-benchmark \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# NGC CLI (version may change)
ARG NGC_CLI_VERSION=4.18.0
RUN wget -q -O /tmp/ngccli.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/${NGC_CLI_VERSION}/files/ngccli_linux.zip && \
    unzip -q /tmp/ngccli.zip -d /opt && rm /tmp/ngccli.zip
ENV PATH="/opt/ngc-cli:${PATH}"
