# Isaac ROS Detection Validation

Offline numeric sanity checks for `vision_msgs/msg/Detection2DArray` output bags.

The main use case is comparing results generated on different machines, for
example NVIDIA reference output vs AMD candidate output. Each machine should run
the same input bag and record only the detection output topic. The two output
bags can then be copied to one machine and compared offline.

Record outputs on each machine:

```bash
ros2 bag record /detections_output -o nv_rtdetr_output
ros2 bag record /detections_output -o amd_rtdetr_output
```

Compare after copying both bags to the same machine:

```bash
ros2 run isaac_ros_detection_validation compare_detection2d_bags.py \
  --reference-bag ./nv_output_bag \
  --candidate-bag ./amd_output_bag \
  --reference-topic /detections_output \
  --candidate-topic /detections_output \
  --match-policy index \
  --output-json /tmp/detection_nv_vs_amd.json
```

Use `--match-policy stamp` when both outputs preserve the same input header
timestamps. Use `--match-policy index` when comparing bags recorded on different
machines where timestamps are not expected to match but message order is.

Useful options:

- `--min-score 0.3`: drop low-confidence detections before comparing.
- `--max-detections-per-frame 100`: keep only the top-scoring detections per frame.
- `--min-frame-pass-rate 0.7`: require this fraction of all evaluated frames to pass.
- `--max-frame-details 20`: write the worst per-frame comparisons into the JSON report.

Unpaired frames count against the overall frame pass rate. This means missing
messages on either side are treated as validation failures, even if all paired
frames look good.
