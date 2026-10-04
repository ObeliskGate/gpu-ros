#!/usr/bin/env python3
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

"""
Compare detections received on two ROS topics.

Messages are paired by header stamp. Detections in paired frames are matched
greedily by IoU; the resulting metrics and coverage counts are observations,
not acceptance thresholds.
"""

import argparse
import math
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
    """
    Greedy IoU matching for one paired frame.

    Returns IoU and score-delta observations, unmatched counts for each input,
    and class-match observations for geometrically matched detections.
    """
    if not dets_a or not dets_b:
        return [], [], len(dets_a), len(dets_b), 0, 0, 0

    pairs = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            pairs.append((iou(to_xyxy(da), to_xyxy(db)), i, j))
    pairs.sort(reverse=True)

    used_a, used_b = set(), set()
    ious, score_deltas = [], []
    class_matches = 0
    class_mismatches = 0
    class_unavailable = 0
    for pair_iou, i, j in pairs:
        if pair_iou <= 0:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(pair_iou)
        if dets_a[i].results and dets_b[j].results:
            result_a = dets_a[i].results[0].hypothesis
            result_b = dets_b[j].results[0].hypothesis
            score_deltas.append(abs(result_a.score - result_b.score))
            if result_a.class_id == result_b.class_id:
                class_matches += 1
            else:
                class_mismatches += 1
        else:
            class_unavailable += 1
    unmatched_a = len(dets_a) - len(used_a)
    unmatched_b = len(dets_b) - len(used_b)
    return (
        ious,
        score_deltas,
        unmatched_a,
        unmatched_b,
        class_matches,
        class_mismatches,
        class_unavailable,
    )


def validate_detection_array(msg):
    """Reject malformed numeric observations before they contaminate metrics."""
    nanoseconds = msg.header.stamp.nanosec
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f'invalid stamp nanoseconds: {nanoseconds}')

    for index, det in enumerate(msg.detections):
        values = (
            det.bbox.center.position.x,
            det.bbox.center.position.y,
            det.bbox.size_x,
            det.bbox.size_y,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f'detection {index} has a non-finite bounding box')
        if det.bbox.size_x < 0 or det.bbox.size_y < 0:
            raise ValueError(f'detection {index} has a negative bounding-box size')
        if det.results and not math.isfinite(float(det.results[0].hypothesis.score)):
            raise ValueError(f'detection {index} has a non-finite score')


def metric_summary(values):
    """Summarize real samples, leaving metrics unavailable when none exist."""
    if not values:
        return {'samples': 0, 'mean': None, 'median': None, 'p95': None}
    samples = np.asarray(values, dtype=float)
    return {
        'samples': int(samples.size),
        'mean': float(samples.mean()),
        'median': float(np.median(samples)),
        'p95': float(np.percentile(samples, 95)),
    }


def format_metric(metric):
    if metric['samples'] == 0:
        return 'n/a'
    return f"mean={metric['mean']:.4f} median={metric['median']:.4f} p95={metric['p95']:.4f}"


class DetectionComparator(Node):
    def __init__(self, topic_a, topic_b):
        super().__init__('detection_comparator')
        self.buf_a = {}
        self.buf_b = {}
        self.data_error = None

        self.received_frames_a = 0
        self.received_frames_b = 0
        self.received_detections_a = 0
        self.received_detections_b = 0
        self.paired_frames = 0
        self.paired_detections_a = 0
        self.paired_detections_b = 0
        self.paired_unmatched_a = 0
        self.paired_unmatched_b = 0
        self.unpaired_frames_a = 0
        self.unpaired_frames_b = 0
        self.unpaired_detections_a = 0
        self.unpaired_detections_b = 0
        self.frame_ious = []
        self.frame_score_deltas = []
        self.matched_detections = 0
        self.class_matches = 0
        self.class_mismatches = 0
        self.class_unavailable = 0

        self.create_subscription(
            Detection2DArray, topic_a, lambda m: self._on(m, self.buf_a, self.buf_b), 10
        )
        self.create_subscription(
            Detection2DArray, topic_b, lambda m: self._on(m, self.buf_b, self.buf_a), 10
        )

    def _on(self, msg, own_buf, other_buf):
        try:
            validate_detection_array(msg)
        except ValueError as error:
            self.data_error = error
            return

        is_a = own_buf is self.buf_a
        detections = len(msg.detections)
        if is_a:
            self.received_frames_a += 1
            self.received_detections_a += detections
        else:
            self.received_frames_b += 1
            self.received_detections_b += detections

        key = stamp_key(msg)
        if key in other_buf:
            other = other_buf.pop(key)
            self.paired_frames += 1
            if is_a:
                detections_a, detections_b = msg.detections, other.detections
            else:
                detections_a, detections_b = other.detections, msg.detections
            self.paired_detections_a += len(detections_a)
            self.paired_detections_b += len(detections_b)

            # Preserve the existing arrival-order matching behavior.
            (
                ious,
                deltas,
                unmatched_msg,
                unmatched_other,
                class_matches,
                class_mismatches,
                class_unavailable,
            ) = match_frame(msg.detections, other.detections)
            if is_a:
                self.paired_unmatched_a += unmatched_msg
                self.paired_unmatched_b += unmatched_other
            else:
                self.paired_unmatched_b += unmatched_msg
                self.paired_unmatched_a += unmatched_other
            if ious:
                self.frame_ious.append(float(np.mean(ious)))
                self.matched_detections += len(ious)
            if deltas:
                self.frame_score_deltas.append(float(np.mean(deltas)))
            self.class_matches += class_matches
            self.class_mismatches += class_mismatches
            self.class_unavailable += class_unavailable
        else:
            if key in own_buf:
                replaced = own_buf[key]
                if is_a:
                    self.unpaired_frames_a += 1
                    self.unpaired_detections_a += len(replaced.detections)
                else:
                    self.unpaired_frames_b += 1
                    self.unpaired_detections_b += len(replaced.detections)
            own_buf[key] = msg

    def summary(self):
        pending_frames_a = len(self.buf_a)
        pending_frames_b = len(self.buf_b)
        pending_detections_a = sum(len(msg.detections) for msg in self.buf_a.values())
        pending_detections_b = sum(len(msg.detections) for msg in self.buf_b.values())
        unpaired_frames_a = self.unpaired_frames_a + pending_frames_a
        unpaired_frames_b = self.unpaired_frames_b + pending_frames_b
        unpaired_detections_a = self.unpaired_detections_a + pending_detections_a
        unpaired_detections_b = self.unpaired_detections_b + pending_detections_b
        class_samples = self.class_matches + self.class_mismatches
        return {
            'frames': {
                'received_a': self.received_frames_a,
                'received_b': self.received_frames_b,
                'paired': self.paired_frames,
                'unmatched_a': unpaired_frames_a,
                'unmatched_b': unpaired_frames_b,
                'pending_a': pending_frames_a,
                'pending_b': pending_frames_b,
            },
            'detections': {
                'received_a': self.received_detections_a,
                'received_b': self.received_detections_b,
                'paired_a': self.paired_detections_a,
                'paired_b': self.paired_detections_b,
                'matched': self.matched_detections,
                'unmatched_paired_a': self.paired_unmatched_a,
                'unmatched_paired_b': self.paired_unmatched_b,
                'unpaired_a': unpaired_detections_a,
                'unpaired_b': unpaired_detections_b,
                'unmatched_a': self.paired_unmatched_a + unpaired_detections_a,
                'unmatched_b': self.paired_unmatched_b + unpaired_detections_b,
                'pending_a': pending_detections_a,
                'pending_b': pending_detections_b,
            },
            'iou': metric_summary(self.frame_ious),
            'score_delta': metric_summary(self.frame_score_deltas),
            'class_observations': {
                'matches': self.class_matches,
                'mismatches': self.class_mismatches,
                'unavailable': self.class_unavailable,
                'samples': class_samples,
                'match_rate': (self.class_matches / class_samples if class_samples else None),
            },
        }

    def report(self):
        summary = self.summary()
        frames = summary['frames']
        detections = summary['detections']
        classes = summary['class_observations']
        class_rate = f'{classes["match_rate"]:.4f}' if classes['match_rate'] is not None else 'n/a'
        print(
            f"Frame coverage: received A={frames['received_a']} B={frames['received_b']} "
            f"paired={frames['paired']} unmatched A={frames['unmatched_a']} "
            f"B={frames['unmatched_b']} "
            f"(pending buffers A={frames['pending_a']} B={frames['pending_b']})"
        )
        print(f"Detections received: A={detections['received_a']} B={detections['received_b']}")
        print(
            f"Paired detections: A={detections['paired_a']} B={detections['paired_b']} "
            f"matched={detections['matched']}"
        )
        print(
            f"Unmatched detections: A={detections['unmatched_a']} "
            f"B={detections['unmatched_b']} "
            f"(paired A={detections['unmatched_paired_a']} "
            f"B={detections['unmatched_paired_b']}; "
            f"unpaired A={detections['unpaired_a']} B={detections['unpaired_b']}; "
            f"pending buffers A={detections['pending_a']} "
            f"B={detections['pending_b']})"
        )
        print(
            f"IoU paired-frame samples={summary['iou']['samples']} {format_metric(summary['iou'])}"
        )
        print(
            f"Score delta paired-frame samples={summary['score_delta']['samples']} "
            f"{format_metric(summary['score_delta'])}"
        )
        print(
            f"Class observations: matches={classes['matches']} "
            f"mismatches={classes['mismatches']} unavailable={classes['unavailable']} "
            f"match_rate={class_rate} samples={classes['samples']}"
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description='Compare two detection topics.')
    parser.add_argument('--topic-a', default='/a/detections_output')
    parser.add_argument('--topic-b', default='/d/detections_output')
    parser.add_argument(
        '--duration', type=float, default=30.0, help='Seconds to collect before reporting'
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration must be a finite, non-negative number')

    node = None
    try:
        rclpy.init()
        node = DetectionComparator(args.topic_a, args.topic_b)
        end = node.get_clock().now().nanoseconds + int(args.duration * 1e9)
        while node.get_clock().now().nanoseconds < end:
            if not rclpy.ok():
                raise RuntimeError('ROS context shut down before collection completed')
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.data_error is not None:
                raise ValueError(f'invalid detection message: {node.data_error}')
        if not rclpy.ok():
            raise RuntimeError('ROS context shut down before reporting')
        node.report()
        return 0
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
