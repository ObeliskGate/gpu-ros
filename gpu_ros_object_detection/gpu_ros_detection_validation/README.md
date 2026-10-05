# gpu_ros_detection_validation

Fixed-input capture, offline `vision_msgs/msg/Detection2DArray` comparison,
transport audits, and AMD benchmark-matrix orchestration. The package consumes
the existing detection graphs; it does not define a second playback framework
or model implementation.

## Installed commands

Run these with `ros2 run gpu_ros_detection_validation <command>` after building
and sourcing the selected runtime:

| Command | Purpose |
| --- | --- |
| `compare_detection2d_bags.py` | Compare recorded detection outputs. |
| `run_amd_phase2a_fixed_input_capture.sh` | Capture AMD RT-DETR or YOLOv8 output. |
| `run_nvidia_fixed_input_capture.sh` | Capture one NVIDIA reference, standard, or managed lane. |
| `run_amd_transport_audit.sh` | Audit AMD standard versus managed transport. |
| `run_nvidia_transport_audit.sh` | Audit NVIDIA Config C versus managed transport. |
| `run_nvidia_yolov8_transport_audit.sh` | Existing YOLOv8-specific audit entry point. |
| `run_nvidia_c_vs_d_copy_audit.sh` | Separate NVIDIA C/D copy diagnostic. |
| `run_rocprof_hip_copy_probe.sh` | Check HIP profiler copy coverage with a supplied test executable. |
| `run_amd_phase2b_benchmark_matrix.sh` | Run the explicitly requested three-round AMD performance matrix. |

## Methods and evidence

[Detection validation methods](../../skills/gpu-ros-experiments/references/detection-validation.md)
contain runtime prerequisites, capture parameters, comparator usage, audits,
HIP probing, and lifecycle checks. [Acceptance rules](../../skills/gpu-ros-experiments/references/result-acceptance.md)
separate detection observations, input/model validity, lifecycle, copy evidence,
and performance.

NVIDIA's migration target is the user-supplied official Isaac ROS 5.0 image
with its paired ROS 2 Lyrical packages. Set `GPU_ROS_NVIDIA_PROFILE=1` before
package discovery; AMD and CPU use `0`. Native transport is named
`tensor_list`, not `nitros`. Native adapter acceptance runs
`test_tensor_list_buffer_adapter` in the compat package build directory;
ORT binding acceptance runs `test_onnx_inference_core` in the ONNX package
build directory. Both are required, with missing tests treated as errors.
These target-runtime checks remain blocked until the authoritative image,
matching dependencies, and GPU runtime are available.

Keep bags, traces, profiles, logs, and reports outside Git. A wrapper exit of
zero does not establish clean component shutdown. Offline bag comparisons use
schema `phase2b_detection_report_only_v2`: `REPORT_ONLY` means a report was
generated, not that numeric outputs are equal. Numeric differences and
unpaired coverage do not set project pass/fail; invalid input or report-writing
errors remain nonzero, as do independent lifecycle and copy failures.

Schema v2 removes numerical thresholds and per-frame/aggregate pass fields.
Consumers should read metric sample counts and coverage rather than a numerical
verdict. No-sample metrics are `null`; a generated report with zero paired
frames does not establish numerical similarity. The CLI has no separate
report-only switch or numerical gate parameters.
