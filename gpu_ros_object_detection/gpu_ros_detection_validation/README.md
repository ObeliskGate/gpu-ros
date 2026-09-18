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
separate correctness from copy evidence and performance.

Keep bags, traces, profiles, logs, and reports outside Git. A wrapper exit of
zero does not establish clean component shutdown, and a REPORT_ONLY comparison
does not pass a correctness gate.
