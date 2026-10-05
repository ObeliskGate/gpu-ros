# Copyright 2026 GPU ROS contributors
# Licensed under the Apache License, Version 2.0.

import os
from pathlib import Path
import subprocess


def test_rejects_multiline_lock_instead_of_ignoring_later_packages():
    installer = Path(__file__).resolve().parents[1] / 'docker/install_isaac_ros_apt_packages.sh'
    # Syntax-only fixture: --check-inputs must never install or fetch packages.
    environment = {
        **os.environ,
        'ROS_DISTRO': 'lyrical',
        'ISAAC_ROS_BASE_IMAGE': 'nvcr.io/nvidia/isaac/ros:syntax-fixture@sha256:' + '0' * 64,
        'ROS_LYRICAL_APT_PACKAGE_SPECS': (
            'ros-lyrical-rclcpp=1\nros-lyrical-isaac-ros-tensor-list-interfaces=2'
        ),
    }
    result = subprocess.run(
        ['bash', str(installer), '--check-inputs'],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
