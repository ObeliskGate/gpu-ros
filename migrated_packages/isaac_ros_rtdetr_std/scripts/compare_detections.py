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
for each paired frame matches detections via IoU. Reports mean/median/p95 IoU,
score delta, unmatched detections, and PASS/FAIL against configurable thresholds.
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

    Returns (matched_ious, matched_score_deltas, unmatched_count). Greedy
    (highest-IoU-first) is sufficient here because the compared pipelines run
    the same model on the same input, so boxes nearly coincide and greedy
    matches the optimal assignment.
    """
    if not dets_a and not dets_b:
        return [], [], 0
    if not dets_a or not dets_b:
        return [], [], max(len(dets_a), len(dets_b))

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
    unmatched_count = (len(dets_a) - len(used_a)) + (len(dets_b) - len(used_b))
    return ious, score_deltas, unmatched_count


class DetectionComparator(Node):

    def __init__(
        self,
        topic_a,
        topic_b,
        min_mean_iou,
        max_mean_score_delta,
        min_frame_pass_rate,
        min_paired_frames,
    ):
        super().__init__('detection_comparator')
        self.min_mean_iou = min_mean_iou
        self.max_mean_score_delta = max_mean_score_delta
        self.min_frame_pass_rate = min_frame_pass_rate
        self.min_paired_frames = min_paired_frames
        self.buf_a = {}
        self.buf_b = {}
        self.frame_ious = []      # mean IoU per paired frame
        self.frame_deltas = []    # mean score delta per paired frame
        self.frame_unmatched = []
        self.frame_passes = []
        self.create_subscription(
            Detection2DArray, topic_a, lambda m: self._on(m, self.buf_a, self.buf_b), 10)
        self.create_subscription(
            Detection2DArray, topic_b, lambda m: self._on(m, self.buf_b, self.buf_a), 10)

    def _on(self, msg, own_buf, other_buf):
        key = stamp_key(msg)
        if key in other_buf:
            other = other_buf.pop(key)
            ious, deltas, unmatched_count = match_frame(
                list(msg.detections), list(other.detections))
            if not ious and unmatched_count == 0:
                self.frame_ious.append(1.0)
                self.frame_deltas.append(0.0)
                self.frame_unmatched.append(0)
                self.frame_passes.append(True)
            elif not ious:
                self.frame_ious.append(0.0)
                self.frame_deltas.append(float('inf'))
                self.frame_unmatched.append(unmatched_count)
                self.frame_passes.append(False)
            else:
                mean_iou = float(np.mean(ious))
                mean_delta = float(np.mean(deltas))
                self.frame_ious.append(mean_iou)
                self.frame_deltas.append(mean_delta)
                self.frame_unmatched.append(unmatched_count)
                self.frame_passes.append(
                    unmatched_count == 0 and
                    mean_iou >= self.min_mean_iou and
                    mean_delta <= self.max_mean_score_delta)
        else:
            own_buf[key] = msg

    def report(self):
        if not self.frame_ious:
            print('No paired frames received — cannot compare.')
            return False
        ious = np.array(self.frame_ious)
        deltas = np.array(self.frame_deltas)
        unmatched = np.array(self.frame_unmatched)
        finite_deltas = deltas[np.isfinite(deltas)]
        delta_mean = finite_deltas.mean() if finite_deltas.size else float('inf')
        delta_median = np.median(finite_deltas) if finite_deltas.size else float('inf')
        delta_p95 = np.percentile(finite_deltas, 95) if finite_deltas.size else float('inf')
        frame_pass = np.mean(np.array(self.frame_passes, dtype=bool))
        unmatched_frames = int(np.count_nonzero(unmatched))
        print(f'Paired frames: {len(ious)}')
        print(f'IoU   mean={ious.mean():.4f} median={np.median(ious):.4f} '
              f'p95={np.percentile(ious, 95):.4f}')
        print(f'Score mean={delta_mean:.4f} median={delta_median:.4f} '
              f'p95={delta_p95:.4f}')
        print(f'Unmatched detections: total={int(unmatched.sum())} '
              f'frames={unmatched_frames}/{len(unmatched)}')
        print(f'Frames passing per-frame threshold: {frame_pass * 100:.1f}%')
        print('Thresholds: '
              f'min_paired_frames={self.min_paired_frames}, '
              f'min_mean_iou={self.min_mean_iou:.4f}, '
              f'max_mean_score_delta={self.max_mean_score_delta:.4f}, '
              f'min_frame_pass_rate={self.min_frame_pass_rate:.4f}')
        ok = (
            len(ious) >= self.min_paired_frames and
            ious.mean() >= self.min_mean_iou and
            delta_mean <= self.max_mean_score_delta and
            frame_pass >= self.min_frame_pass_rate)
        print('PASS' if ok else 'FAIL')
        return ok


def main():
    parser = argparse.ArgumentParser(description='Compare two detection topics.')
    parser.add_argument('--topic-a', default='/a/detections_output')
    parser.add_argument('--topic-b', default='/d/detections_output')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Seconds to collect before reporting')
    parser.add_argument('--min-mean-iou', type=float, default=0.95,
                        help='Minimum mean IoU across paired frames')
    parser.add_argument('--max-mean-score-delta', type=float, default=0.05,
                        help='Maximum mean score delta across paired frames')
    parser.add_argument('--min-frame-pass-rate', type=float, default=0.90,
                        help='Minimum fraction of paired frames passing per-frame thresholds')
    parser.add_argument('--min-paired-frames', type=int, default=1,
                        help='Minimum number of paired frames required')
    args = parser.parse_args()

    rclpy.init()
    node = DetectionComparator(
        args.topic_a,
        args.topic_b,
        args.min_mean_iou,
        args.max_mean_score_delta,
        args.min_frame_pass_rate,
        args.min_paired_frames,
    )
    end = node.get_clock().now().nanoseconds + int(args.duration * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = node.report()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
