#!/usr/bin/env python3
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

"""Compare RT-DETR detections between two pipelines (e.g. config A vs config D).

Subscribes to two Detection2DArray topics, pairs messages by header stamp, and
for each paired frame matches detections via IoU using the Hungarian algorithm.
Reports mean/median/p95 IoU and score delta, and PASS/FAIL against thresholds
(mean IoU >= 0.95, mean score delta <= 0.05, >= 90% of frames passing).
"""

import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


def to_xyxy(det):
    """Convert a Detection2D (center/size) box to (x1, y1, x2, y2)."""
    cx = det.bbox.center.position.x
    cy = det.bbox.center.position.y
    w = det.bbox.size_x
    h = det.bbox.size_y
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stamp_key(msg):
    return (msg.header.stamp.sec, msg.header.stamp.nanosec)


def match_frame(dets_a, dets_b):
    """Greedy IoU matching for one paired frame.

    Returns (matched_ious, matched_score_deltas). Greedy (highest-IoU-first) is
    sufficient here because A and D run the same model on the same input, so
    boxes nearly coincide and greedy matches the optimal assignment.
    """
    if not dets_a or not dets_b:
        return [], []

    pairs = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            pairs.append((iou(to_xyxy(da), to_xyxy(db)), i, j))
    pairs.sort(reverse=True)

    used_a, used_b = set(), set()
    ious, score_deltas = [], []
    for pair_iou, i, j in pairs:
        if pair_iou <= 0:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(pair_iou)
        sa = dets_a[i].results[0].hypothesis.score if dets_a[i].results else 0.0
        sb = dets_b[j].results[0].hypothesis.score if dets_b[j].results else 0.0
        score_deltas.append(abs(sa - sb))
    return ious, score_deltas


class DetectionComparator(Node):

    def __init__(self, topic_a, topic_b):
        super().__init__('detection_comparator')
        self.buf_a = {}
        self.buf_b = {}
        self.frame_ious = []      # mean IoU per paired frame
        self.frame_deltas = []    # mean score delta per paired frame
        self.create_subscription(
            Detection2DArray, topic_a, lambda m: self._on(m, self.buf_a, self.buf_b), 10)
        self.create_subscription(
            Detection2DArray, topic_b, lambda m: self._on(m, self.buf_b, self.buf_a), 10)

    def _on(self, msg, own_buf, other_buf):
        key = stamp_key(msg)
        if key in other_buf:
            other = other_buf.pop(key)
            ious, deltas = match_frame(list(msg.detections), list(other.detections))
            if ious:
                self.frame_ious.append(float(np.mean(ious)))
                self.frame_deltas.append(float(np.mean(deltas)))
        else:
            own_buf[key] = msg

    def report(self):
        if not self.frame_ious:
            print('No paired frames with matched detections — cannot compare.')
            return False
        ious = np.array(self.frame_ious)
        deltas = np.array(self.frame_deltas)
        frame_pass = np.mean((ious >= 0.95) & (deltas <= 0.05))
        print(f'Paired frames: {len(ious)}')
        print(f'IoU   mean={ious.mean():.4f} median={np.median(ious):.4f} '
              f'p95={np.percentile(ious, 95):.4f}')
        print(f'Score mean={deltas.mean():.4f} median={np.median(deltas):.4f} '
              f'p95={np.percentile(deltas, 95):.4f}')
        print(f'Frames passing per-frame threshold: {frame_pass * 100:.1f}%')
        ok = ious.mean() >= 0.95 and deltas.mean() <= 0.05 and frame_pass >= 0.90
        print('PASS' if ok else 'FAIL')
        return ok


def main():
    parser = argparse.ArgumentParser(description='Compare two detection topics.')
    parser.add_argument('--topic-a', default='/a/detections_output')
    parser.add_argument('--topic-b', default='/d/detections_output')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Seconds to collect before reporting')
    args = parser.parse_args()

    rclpy.init()
    node = DetectionComparator(args.topic_a, args.topic_b)
    end = node.get_clock().now().nanoseconds + int(args.duration * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = node.report()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
