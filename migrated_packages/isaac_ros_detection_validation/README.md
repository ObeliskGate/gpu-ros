# Isaac ROS Detection Validation

Offline numeric sanity checks for `vision_msgs/msg/Detection2DArray` output bags.

The main use case is comparing results generated on different machines, for
example NVIDIA reference output vs AMD candidate output. Each machine should run
the same input bag and record only the detection output topic. The two output
bags can then be copied to one machine and compared offline.

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
