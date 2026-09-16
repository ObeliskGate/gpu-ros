# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_fixed_input_capture.sh'
AMD_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2a_fixed_input_capture.sh'
AUDIT_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_yolov8_transport_audit.sh'
UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_transport_audit.sh'
)
NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_c_vs_d_copy_audit.sh'
)
UNIFIED_AMD_AUDIT_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_amd_transport_audit.sh'
ROCPROF_HIP_PROBE_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_rocprof_hip_copy_probe.sh'
)


def test_capture_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<rtdetr-c|rtdetr-d|rtdetr-managed|yolov8-c|yolov8-d|yolov8-managed>' in (result.stdout)


def test_capture_runner_rejects_an_unknown_lane_before_starting_ros():
    result = subprocess.run(
        [str(SCRIPT_PATH), 'unknown', 'capture_name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'unsupported lane' in result.stderr
    assert 'unknown' in result.stderr


def test_amd_capture_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'CAPTURE_TRANSPORT=std|managed' in result.stdout


def test_amd_capture_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'output-name' in result.stderr


def test_amd_yolov8_capture_fails_before_ros_when_asset_is_missing(tmp_path):
    environment = os.environ.copy()
    environment['OVG_ASSETS_ROOT'] = str(tmp_path / 'assets')
    environment['CAPTURE_INPUT_BAG'] = str(tmp_path / 'missing-bag')
    model_path = tmp_path / 'assets' / 'models' / 'yolov8' / 'yolov8s.onnx'
    environment['CAPTURE_MODEL_PATH'] = str(model_path)
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'yolov8', 'missing-model'],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert 'missing' in result.stderr
    assert str(model_path) in result.stderr


def test_nvidia_c_vs_d_audit_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<yolov8|rtdetr> <audit-name>' in result.stdout


def test_unified_audit_help_does_not_require_ros_environment():
    for script in (UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH, UNIFIED_AMD_AUDIT_SCRIPT_PATH):
        result = subprocess.run([str(script), '--help'], check=True, capture_output=True, text=True)
        assert '<yolov8|rtdetr> <audit-name>' in result.stdout


def test_transport_audit_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(AUDIT_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<audit-name>' in result.stdout


def test_transport_audit_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AUDIT_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'audit-name' in result.stderr


def test_rocprof_hip_probe_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(ROCPROF_HIP_PROBE_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<hip-copy-executable> <output-directory>' in result.stdout
