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
AMD_MATRIX_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2b_benchmark_matrix.sh'
)
AMD_HIGH_LOAD_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2b_managed_high_load.sh'
)
AMD_PHASE2_LAUNCHER_PATH = Path(__file__).parents[3] / 'docker' / 'phase2-amd.sh'
HIP_PREPROCESS_SOURCE_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_detection_common' / 'src' / 'hip_preprocess.hip'
)
DETECTION_COMMON_CMAKE_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_detection_common' / 'CMakeLists.txt'
)
YOLOV8_CMAKE_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_yolov8' / 'CMakeLists.txt'
)
RTDETR_CMAKE_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_rtdetr' / 'CMakeLists.txt'
)
YOLOV8_MANAGED_HEADER_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_yolov8' / 'include' / 'gpu_ros_yolov8' /
    'yolov8_managed_hip_nodes.hpp'
)
RTDETR_MANAGED_HEADER_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_rtdetr' / 'include' / 'gpu_ros_rtdetr' /
    'rtdetr_managed_hip_nodes.hpp'
)
ONNX_INFERENCE_CORE_SOURCE_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' /
    'gpu_ros_onnx_inference' / 'src' / 'onnx_inference_core.cpp'
)
MANAGED_HIP_POL_PATH = (
    Path(__file__).parents[3] / 'migrated_packages' / 'gpu_ros_onnx_inference' /
    'test' / 'gpu_ros_onnx_managed_hip_pol_test.py'
)
ROCPROF_HIP_PROBE_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_rocprof_hip_copy_probe.sh'
)
AMD_COLCON_DEFAULTS_PATH = (
    Path(__file__).parents[3] / 'docker' / 'colcon-defaults-phase2a-amd.yaml'
)
BENCHMARKS_ROOT = Path(__file__).parents[3] / 'migrated_packages' / 'benchmarks'
REPOSITORY_ROOT = Path(__file__).parents[3]


def test_staged_control_metadata_matches_component_topology():
    for name in (
            'gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py',
            'gpu_ros_yolov8_phase2b_amd_staged_control_graph.py'):
        source = (BENCHMARKS_ROOT / name).read_text()
        assert 'Managed->std->Managed before ORT; ' in source
        assert 'Managed->std->standard decoder after ORT' in source
        assert source.index('managed_to_std_input =') < source.index('std_to_managed_input =')
        assert source.index('std_to_managed_input =') < source.index('onnx =')
        assert source.index('onnx =') < source.index('managed_to_std_output =')
        assert source.index('managed_to_std_output =') < source.index('decoder =')
        assert 'std_to_managed_output =' not in source


def test_benchmark_manifest_records_complete_dirty_repository_state():
    source = AMD_MATRIX_SCRIPT_PATH.read_text()
    assert 'git -C "${path}" diff HEAD --binary' in source
    assert 'ls-files --others --exclude-standard' in source
    assert 'application_untracked_paths=' in source
    assert 'application_untracked_content_sha256=' in source
    assert 'gpu_ros_managed_diff_head_binary_sha256=' in source
    assert 'gpu_ros_managed_untracked_content_sha256=' in source
    assert 'submodule_status' not in source


def test_nvidia_sources_are_opt_in_external_checkouts():
    assert not (REPOSITORY_ROOT / '.gitmodules').exists()
    manifest = (REPOSITORY_ROOT / 'external' / 'nvidia-isaac-ros.repos').read_text()
    assert '060ced887bd8a3a0be60b1fa454365942eefd128' in manifest
    assert 'f46699e124262c5bfb6f00099061f6718f026b3f' in manifest
    amd_dockerfile = (REPOSITORY_ROOT / 'docker' / 'phase2a-amd.Dockerfile').read_text()
    assert 'bootstrap-nvidia-external' not in amd_dockerfile
    assert 'isaac_ros_object_detection.git' not in amd_dockerfile


def test_ort_lock_is_the_only_version_and_commit_source():
    lock = (REPOSITORY_ROOT / 'config' / 'onnxruntime.lock').read_text()
    assert 'ORT_VERSION=1.23.1' in lock
    assert 'ORT_COMMIT=d9b2048791efb5804fe3d53a04b4971256addebf' in lock
    builder = (REPOSITORY_ROOT / 'tools' / 'build-phase2a-external-ort.sh').read_text()
    assert 'source "${ORT_LOCK}"' in builder
    assert '"${source_commit}" == "${ORT_COMMIT}"' in builder


def test_cpu_defaults_and_offline_asset_profiles_are_explicit():
    cmake = (
        REPOSITORY_ROOT / 'migrated_packages' / 'gpu_ros_onnx_inference' /
        'CMakeLists.txt').read_text()
    assert 'option(ORT_ENABLE_CUDA "Enable CUDAExecutionProvider selection" OFF)' in cmake
    assert 'option(BUILD_NITROS_TRANSPORT' in cmake
    assert 'NVIDIA Isaac ROS)" OFF)' in cmake
    assets = (REPOSITORY_ROOT / 'tools' / 'phase2-assets').read_text()
    assert 'urllib' not in assets
    assert 'nvidia_synthetica" if provider == "cuda" else "rtdetrv2_r50"' in assets
    assert 'automatic fallback is forbidden' in assets


def test_nitros_conversion_callbacks_drop_exceptions():
    source = (
        REPOSITORY_ROOT / 'migrated_packages' / 'gpu_ros_onnx_inference' /
        'src' / 'nitros_managed_tensor_bundle_nodes.cpp').read_text()
    assert source.count('catch (const std::exception & error)') >= 2
    assert source.count('catch (...)') >= 2
    io_source = (
        REPOSITORY_ROOT / 'migrated_packages' / 'gpu_ros_onnx_inference' /
        'src' / 'nitros_tensor_bundle_io.cpp').read_text()
    assert 'Dropping NITROS input frame' in io_source
    assert 'Dropping NITROS output frame' in io_source


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


def test_nvidia_capture_runner_uses_compat_package_for_nitros_launches():
    script = SCRIPT_PATH.read_text()
    assert 'LAUNCH_PACKAGE="gpu_ros_nvidia_tensor_bundle_compat"' in script
    assert script.count('LAUNCH_PACKAGE="gpu_ros_nvidia_tensor_bundle_compat"') == 3
    assert 'LAUNCH_PACKAGE="gpu_ros_onnx_inference"' not in script


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
    assert '--disable-keyboard-controls' in script
    assert '/rosbag2_recorder/stop' in script


def test_amd_launcher_discovers_persistent_content_named_sif():
    """Use the repository-local .ovg state and content-named SIF discovery."""
    script = AMD_PHASE2_LAUNCHER_PATH.read_text()
    subprocess.run(['bash', '-n', str(AMD_PHASE2_LAUNCHER_PATH)], check=True)
    assert 'STATE_ROOT="${OVG_STATE_ROOT:-${ROOT_DIR}/.ovg}"' in script
    assert 'ROOT_DIR}/../ovg-state' not in script
    assert "-name 'phase2-amd*.sif'" in script
    assert 'multiple Apptainer SIFs found' in script
    assert 'No Apptainer SIF found under' in script


def test_amd_launcher_requires_external_ort_for_formal_runs():
    script = AMD_PHASE2_LAUNCHER_PATH.read_text()
    assert 'resolve_external_ort' in script
    assert 'require_external_ort' in script
    assert 'legacy build-only content' in script
    assert 'bootstrap|up|colcon|verify' in script


def test_amd_launcher_keeps_device_gate_when_slurm_is_warning_only():
    """Slurm policy is optional, while real AMD device visibility is not."""
    script = AMD_PHASE2_LAUNCHER_PATH.read_text()
    assert 'require_explicit_apptainer_environment' in script
    assert 'require_slurm_compute_node' in script
    assert 'SLURM_JOB_ID' in script
    assert 'squeue -h -j' in script
    assert '/dev/kfd' in script
    assert '/dev/dri' in script
    assert 'OVG_REQUIRE_SLURM' in script
    assert 'local require_slurm="${OVG_REQUIRE_SLURM:-0}"' in script
    assert 'set OVG_REQUIRE_SLURM=1 to make this fatal' in script
    assert 'scontrol show hostnames' in script
    assert 'OVG_REQUIRE_SLURM=0 explicitly disables' not in script
    assert 'OVG_ALLOW_BUILD_ONLY_SHELL' in script
    assert 'preflight' in script


def test_hip_preprocess_source_includes_full_runtime_header():
    script = HIP_PREPROCESS_SOURCE_PATH.read_text()
    assert '#include <hip/hip_runtime.h>' in script
    assert 'hipLaunchKernelGGL' in script
    assert 'blockIdx' in script


def test_strict_ort_outputs_are_mutable_for_pointer_identity_check():
    source = ONNX_INFERENCE_CORE_SOURCE_PATH.read_text()
    assert 'auto ort_outputs = binding.GetOutputValues();' in source
    assert 'const auto ort_outputs = binding.GetOutputValues();' not in source


def test_detection_common_exports_namespaced_targets_to_downstream_packages():
    common = DETECTION_COMMON_CMAKE_PATH.read_text()
    yolov8 = YOLOV8_CMAKE_PATH.read_text()
    rtdetr = RTDETR_CMAKE_PATH.read_text()
    assert 'NAMESPACE gpu_ros_detection_common::' in common
    assert 'EXPORT_NAME core' in common
    assert 'EXPORT_NAME hip' in common
    assert 'gpu_ros_detection_commonHipTargets' not in common
    assert common.count('EXPORT gpu_ros_detection_commonTargets') == 2
    assert 'gpu_ros_detection_common::core' in yolov8
    assert 'gpu_ros_detection_common::hip' in yolov8
    assert 'gpu_ros_detection_common::core' in rtdetr
    assert 'gpu_ros_detection_common::hip' in rtdetr


def test_managed_tensor_publishers_include_the_tensor_list_type_adapter():
    for header_path in (YOLOV8_MANAGED_HEADER_PATH, RTDETR_MANAGED_HEADER_PATH):
        header = header_path.read_text()
        assert 'gpu_ros_managed_tensor_bundle/type_adapter.hpp' in header


def test_amd_managed_benchmark_graphs_use_migraphx_and_managed_transport():
    """Keep AMD Managed HIP throughput entry points separate from CUDA graphs."""
    benchmark_paths = (
        BENCHMARKS_ROOT / 'gpu_ros_rtdetr_phase2b_amd_managed_graph.py',
        BENCHMARKS_ROOT / 'gpu_ros_yolov8_phase2b_amd_managed_graph.py',
    )
    for benchmark_path in benchmark_paths:
        assert benchmark_path.is_file()
        script = benchmark_path.read_text()
        assert '"execution_provider": "migraphx"' in script
        assert '"transport": "managed"' in script
        assert '"managed_io_contract": "hip_managed_strict"' in script
        assert 'StdToManagedHipTensorBundleNode' not in script
        assert 'ManagedHipToStdTensorBundleNode' not in script
        assert 'YoloV8ManagedHip' in script or 'RtDetrManagedHip' in script
        assert 'execution_provider": "cuda"' not in script


def test_managed_hip_pol_uses_direct_strict_production_plugins():
    script = MANAGED_HIP_POL_PATH.read_text()
    assert 'hip_managed_strict' in script
    assert 'YoloV8ManagedHipImageEncoderNode' in script
    assert 'YoloV8ManagedHipDecoderNode' in script
    assert 'RtDetrManagedHipImageEncoderNode' in script
    assert 'RtDetrManagedHipPreprocessorNode' in script
    assert 'RtDetrManagedHipDecoderNode' in script
    assert 'labels=int64[1,300]' in script
    assert 'boxes=float32[1,300,4]' in script
    assert 'scores=float32[1,300]' in script
    assert 'StdToManagedHipTensorBundleNode' not in script
    assert 'ManagedHipToStdTensorBundleNode' not in script


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
    assert 'Run phase2 assets import-model --profile yolov8 --source <path>' in result.stderr


def test_transport_audit_runner_is_executable_and_has_valid_bash_syntax():
    assert AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AUDIT_SCRIPT_PATH)], check=True)
    assert 'run_nvidia_transport_audit.sh' in AUDIT_SCRIPT_PATH.read_text()


def test_amd_phase2b_matrix_runner_has_rotating_independent_lanes():
    assert AMD_MATRIX_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AMD_MATRIX_SCRIPT_PATH)], check=True)
    script = AMD_MATRIX_SCRIPT_PATH.read_text()
    assert 'round_count=3' in script
    assert 'lane_order_round_1=std,staged,direct' in script
    assert 'lane_order_round_2=staged,direct,std' in script
    assert 'lane_order_round_3=direct,std,staged' in script
    assert 'launch_test' in script


def test_amd_phase2b_high_load_runner_enforces_the_hard_gate():
    assert AMD_HIGH_LOAD_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AMD_HIGH_LOAD_SCRIPT_PATH)], check=True)
    script = AMD_HIGH_LOAD_SCRIPT_PATH.read_text()
    assert 'DURATION_SECONDS < 600' in script
    assert 'MIN_INPUT_MESSAGES < 10000' in script
    assert 'count_ros_messages.py' in script
    assert 'hip_managed_strict' in script


def test_amd_phase2b_high_load_discovers_root_detection_topic():
    script = AMD_HIGH_LOAD_SCRIPT_PATH.read_text()
    assert 'DETECTION_CANDIDATES=(' in script
    assert '/detections_output' in script
    assert 'discover_detection_topic()' in script
    assert 'resolved_detection_topic=' in script


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
    assert 'ROCP_TOOL_ATTACH=1' in script
    assert 'ROCP_TOOL_ATTACH=1 \\\n  CAPTURE_TRANSPORT=' in script
    assert '--report-only' in script
    assert 'Direct Managed HIP production lane expects no application TensorBundle staging copies.' in script
    assert 'fixed_input_playback_pending=true' in (
        Path(__file__).parents[1] / 'scripts' /
        'run_amd_phase2a_fixed_input_capture.sh').read_text()


def test_rocprof_hip_probe_is_explicit_and_requests_json_csv_and_hip_trace():
    assert ROCPROF_HIP_PROBE_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(ROCPROF_HIP_PROBE_SCRIPT_PATH)], check=True)
    result = subprocess.run(
        [str(ROCPROF_HIP_PROBE_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    script = ROCPROF_HIP_PROBE_SCRIPT_PATH.read_text()
    assert '<hip-copy-executable> <output-directory>' in result.stdout
    assert '--memory-copy-trace' in script
    assert '--kernel-trace' in script
    assert '--hip-trace' in script
    assert '--output-format json csv' in script
    assert 'does not assume a build target name' in script


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
    assert 'CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE' in script
    assert 'CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE' in script
    assert 'wait_for_playback_done_file' in script
    assert 'ROCPROF_OUTPUT_PATH=' in script
    assert 'ROCPROF_OUTPUT_FILE_NAME=' in script
    assert 'ROCPROF_OUTPUT_FORMAT=json' in script
    assert 'wait_for_trace_outputs' in script


def test_capture_runner_holds_graph_until_profiler_detaches():
    script = (
        Path(__file__).parents[1] / 'scripts' /
        'run_amd_phase2a_fixed_input_capture.sh'
    ).read_text()
    assert 'CAPTURE_TRACE_ATTACH_PLAYBACK_DONE_FILE' in script
    assert 'CAPTURE_TRACE_ATTACH_DETACH_COMPLETE_FILE' in script
    assert 'wait_for_profiler_detach' in script


def test_amd_profile_excludes_legacy_nvidia_yolov8_pol_test():
    profile = AMD_COLCON_DEFAULTS_PATH.read_text()
    cmake = YOLOV8_CMAKE_PATH.read_text()
    assert '-DBUILD_NVIDIA_YOLOV8_POL_TEST=OFF' in profile
    assert 'test_gpu_ros_std_yolov8_pol_test' in profile
    assert 'NOT BUILD_NITROS_TRANSPORT AND ORT_ENABLE_MIGRAPHX' in cmake
    assert 'set(BUILD_NVIDIA_YOLOV8_POL_TEST OFF CACHE BOOL' in cmake


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
