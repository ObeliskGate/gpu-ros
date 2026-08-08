# Copyright 2026 Maintainer
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

SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_fixed_input_capture.sh'
)
AMD_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2a_fixed_input_capture.sh'
)
AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_yolov8_transport_audit.sh'
)
UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_transport_audit.sh'
)
NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_c_vs_d_copy_audit.sh'
)
UNIFIED_AMD_AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_amd_transport_audit.sh'
)
AMD_COLCON_DEFAULTS_PATH = (
    Path(__file__).parents[3] / 'docker' / 'colcon-defaults-phase2a-amd.yaml'
)


def test_capture_runner_is_executable_and_has_valid_bash_syntax():
    assert SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(SCRIPT_PATH)], check=True)


def test_capture_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<rtdetr-c|rtdetr-d|rtdetr-managed|yolov8-c|yolov8-d|yolov8-managed>' in (
        result.stdout)


def test_capture_runner_rejects_an_unknown_lane_before_starting_ros():
    result = subprocess.run(
        [str(SCRIPT_PATH), 'unknown', 'capture_name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsupported lane 'unknown'" in result.stderr


def test_amd_capture_runner_is_executable_and_has_valid_bash_syntax():
    assert AMD_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AMD_SCRIPT_PATH)], check=True)


def test_amd_capture_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'record AMD Phase 2A in one terminal' in result.stdout
    assert 'CAPTURE_TRANSPORT=std|managed' in result.stdout


def test_amd_capture_runner_has_managed_launches():
    script = AMD_SCRIPT_PATH.read_text()
    assert 'rtdetr_ort_managed_amd.launch.py' in script
    assert 'yolov8_ort_managed_amd.launch.py' in script


def test_amd_capture_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'output-name may contain only' in result.stderr


def test_amd_capture_runner_matches_nvidia_config_c_orig_target_size():
    script = AMD_SCRIPT_PATH.read_text()
    assert 'input_image_width:=1280' in script
    assert 'input_image_height:=720' in script
    assert 'use_max_dim_for_orig_size:=true' in script


def test_amd_yolov8_capture_fails_before_ros_when_asset_is_missing(tmp_path):
    environment = os.environ.copy()
    environment['OVG_ASSETS_ROOT'] = str(tmp_path / 'assets')
    environment['CAPTURE_INPUT_BAG'] = str(tmp_path / 'missing-bag')
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'yolov8', 'missing-model'],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert 'YOLOv8 ONNX asset is missing:' in result.stderr
    assert 'Provide OVG_YOLOV8_ONNX_SOURCE and run phase2 assets import-yolov8.' in result.stderr


def test_transport_audit_runner_is_executable_and_has_valid_bash_syntax():
    assert AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AUDIT_SCRIPT_PATH)], check=True)
    assert 'run_nvidia_transport_audit.sh' in AUDIT_SCRIPT_PATH.read_text()


def test_unified_nvidia_audit_runner_is_executable_and_has_valid_bash_syntax():
    assert UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH)], check=True)
    script = UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH.read_text()
    assert '<yolov8|rtdetr> <audit-name>' in script
    assert '--config-c-binding-report' in script
    assert '--managed-binding-report' in script
    assert 'Config A is a manual sanity reference only' in script


def test_nvidia_c_vs_d_audit_runner_is_executable_and_has_valid_bash_syntax():
    """The C-vs-D experiment has a separate diagnostic runner."""
    assert NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH)], check=True)
    script = NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH.read_text()
    assert 'compare_nvidia_copy_traces.py' in script
    assert 'rtdetr-d' in SCRIPT_PATH.read_text()
    assert 'yolov8-d' in SCRIPT_PATH.read_text()


def test_nvidia_c_vs_d_audit_help_does_not_require_ros_environment():
    """Help is available before sourcing ROS or checking GPU tools."""
    result = subprocess.run(
        [str(NVIDIA_C_VS_D_AUDIT_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<yolov8|rtdetr> <audit-name>' in result.stdout


def test_unified_amd_audit_uses_attach_trace_without_warmup_fallback():
    assert UNIFIED_AMD_AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(UNIFIED_AMD_AUDIT_SCRIPT_PATH)], check=True)
    script = UNIFIED_AMD_AUDIT_SCRIPT_PATH.read_text()
    assert 'rocprofv3 --attach' in script
    assert '--memory-copy-trace' in script
    assert '--kernel-trace' in script
    assert '--output-format json' in script
    assert '--attach-sync-output' in script
    assert 'ROCP_TOOL_ATTACH' not in script
    assert 'fixed_input_playback_pending=true' in (
        Path(__file__).parents[1] / 'scripts' /
        'run_amd_phase2a_fixed_input_capture.sh').read_text()


def test_unified_amd_audit_matches_capture_runner_argument_contract():
    """Use one argument for RT-DETR and two for the YOLOv8 lane."""
    script = UNIFIED_AMD_AUDIT_SCRIPT_PATH.read_text()
    assert 'local capture_args=("${output_name}")' in script
    assert 'capture_args=(yolov8 "${output_name}")' in script
    assert '"${CAPTURE_RUNNER}" "${capture_args[@]}"' in script


def test_unified_amd_audit_handles_rocprof_versions_without_sync_output():
    script = UNIFIED_AMD_AUDIT_SCRIPT_PATH.read_text()
    assert "rocprofv3 --help" in script
    assert 'ROCPROF_ATTACH_ARGS=()' in script
    assert '--attach-duration-msec' in script
    assert 'kill -INT "${PROFILER_PID}"' in script
    assert 'wait_for_trace_outputs' in script


def test_amd_profile_excludes_legacy_nvidia_yolov8_pol_test():
    profile = AMD_COLCON_DEFAULTS_PATH.read_text()
    assert '-DBUILD_NVIDIA_YOLOV8_POL_TEST=OFF' in profile
    assert 'test_isaac_ros_std_yolov8_pol_test' in profile


def test_unified_audit_help_does_not_require_ros_environment():
    for script in (UNIFIED_NVIDIA_AUDIT_SCRIPT_PATH, UNIFIED_AMD_AUDIT_SCRIPT_PATH):
        result = subprocess.run(
            [str(script), '--help'], check=True, capture_output=True, text=True)
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
    assert 'audit-name may contain only' in result.stderr
