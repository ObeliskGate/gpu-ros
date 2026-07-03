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

"""Compare two Detection2DArray output bags."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from vision_msgs.msg import Detection2D, Detection2DArray


DETECTION2D_ARRAY_TYPE = 'vision_msgs/msg/Detection2DArray'


@dataclass(frozen=True)
class DetectionFrame:
    """A Detection2DArray message plus bag/index metadata."""

    index: int
    stamp_ns: int
    bag_time_ns: int
    detections: Sequence[Detection2D]


@dataclass(frozen=True)
class FrameComparison:
    """Comparison metrics for one paired frame."""

    reference_index: int
    candidate_index: int
    mean_iou: float
    mean_score_delta: float
    class_match_rate: float
    matched_count: int
    unmatched_count: int
    passed: bool


@dataclass(frozen=True)
class Thresholds:
    min_mean_iou: float
    max_mean_score_delta: float
    min_frame_pass_rate: float
    min_paired_frames: int
    min_class_match_rate: float


def normalize_topic(topic: str) -> str:
    return '/' + topic.strip('/')


def topic_matches(actual: str, requested: str) -> bool:
    return normalize_topic(actual) == normalize_topic(requested)


def resolve_storage_id(bag_path: str, storage_id: str) -> str:
    if storage_id:
        return storage_id
    metadata_path = Path(bag_path) / 'metadata.yaml'
    if not metadata_path.is_file():
        return ''
    for line in metadata_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('storage_identifier:'):
            return stripped.split(':', 1)[1].strip().strip('"\'')
    return ''


def stamp_key(msg: Detection2DArray) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def detection_score(det: Detection2D) -> float:
    return float(det.results[0].hypothesis.score) if det.results else 0.0


def detection_class_id(det: Detection2D) -> str:
    return str(det.results[0].hypothesis.class_id) if det.results else ''


def to_xyxy(det: Detection2D) -> Tuple[float, float, float, float]:
    cx = float(det.bbox.center.position.x)
    cy = float(det.bbox.center.position.y)
    width = float(det.bbox.size_x)
    height = float(det.bbox.size_y)
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def read_detection_frames(bag_path: str, topic: str, storage_id: str = '') -> List[DetectionFrame]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id=resolve_storage_id(bag_path, storage_id),
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    topic_types = {
        normalize_topic(t.name): t.type
        for t in reader.get_all_topics_and_types()
    }
    normalized_topic = normalize_topic(topic)
    if normalized_topic not in topic_types:
        available = ', '.join(sorted(topic_types))
        raise RuntimeError(
            f"Topic '{topic}' not found in bag '{bag_path}'. Available topics: {available}")
    if topic_types[normalized_topic] != DETECTION2D_ARRAY_TYPE:
        raise RuntimeError(
            f"Topic '{topic}' has type '{topic_types[normalized_topic]}', "
            f"expected '{DETECTION2D_ARRAY_TYPE}'.")

    frames: List[DetectionFrame] = []
    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()
        if not topic_matches(topic_name, topic):
            continue
        msg = deserialize_message(data, Detection2DArray)
        frames.append(
            DetectionFrame(
                index=len(frames),
                stamp_ns=stamp_key(msg),
                bag_time_ns=int(bag_time_ns),
                detections=list(msg.detections),
            )
        )
    return frames


def pair_frames(
    reference: Sequence[DetectionFrame],
    candidate: Sequence[DetectionFrame],
    match_policy: str,
) -> Tuple[List[Tuple[DetectionFrame, DetectionFrame]], int, int]:
    if match_policy == 'index':
        paired_count = min(len(reference), len(candidate))
        pairs = list(zip(reference[:paired_count], candidate[:paired_count]))
        return pairs, len(reference) - paired_count, len(candidate) - paired_count

    if match_policy != 'stamp':
        raise ValueError(f"Unsupported match policy: {match_policy}")

    candidate_by_stamp: Dict[int, Deque[DetectionFrame]] = defaultdict(deque)
    for frame in candidate:
        candidate_by_stamp[frame.stamp_ns].append(frame)

    pairs = []
    unpaired_reference = 0
    for ref_frame in reference:
        queue = candidate_by_stamp.get(ref_frame.stamp_ns)
        if queue:
            pairs.append((ref_frame, queue.popleft()))
        else:
            unpaired_reference += 1

    unpaired_candidate = sum(len(queue) for queue in candidate_by_stamp.values())
    return pairs, unpaired_reference, unpaired_candidate


def match_detections(
    reference_detections: Sequence[Detection2D],
    candidate_detections: Sequence[Detection2D],
) -> Tuple[List[float], List[float], int, int]:
    if not reference_detections and not candidate_detections:
        return [], [], 0, 0
    if not reference_detections or not candidate_detections:
        return [], [], max(len(reference_detections), len(candidate_detections)), 0

    candidate_pairs = []
    for ref_index, ref_det in enumerate(reference_detections):
        ref_box = to_xyxy(ref_det)
        for cand_index, cand_det in enumerate(candidate_detections):
            candidate_pairs.append((iou(ref_box, to_xyxy(cand_det)), ref_index, cand_index))
    candidate_pairs.sort(reverse=True)

    used_reference = set()
    used_candidate = set()
    ious = []
    score_deltas = []
    class_matches = 0

    for pair_iou, ref_index, cand_index in candidate_pairs:
        if pair_iou <= 0.0:
            break
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        ious.append(pair_iou)
        score_deltas.append(
            abs(detection_score(reference_detections[ref_index]) -
                detection_score(candidate_detections[cand_index])))
        if detection_class_id(reference_detections[ref_index]) == detection_class_id(
            candidate_detections[cand_index]
        ):
            class_matches += 1

    unmatched_count = (
        len(reference_detections) - len(used_reference) +
        len(candidate_detections) - len(used_candidate)
    )
    return ious, score_deltas, unmatched_count, class_matches


def compare_frame(
    reference: DetectionFrame,
    candidate: DetectionFrame,
    thresholds: Thresholds,
) -> FrameComparison:
    ious, score_deltas, unmatched_count, class_matches = match_detections(
        reference.detections, candidate.detections)

    if not ious and unmatched_count == 0:
        mean_iou = 1.0
        mean_score_delta = 0.0
        class_match_rate = 1.0
        matched_count = 0
    elif not ious:
        mean_iou = 0.0
        mean_score_delta = float('inf')
        class_match_rate = 0.0
        matched_count = 0
    else:
        mean_iou = float(np.mean(ious))
        mean_score_delta = float(np.mean(score_deltas))
        class_match_rate = class_matches / len(ious)
        matched_count = len(ious)

    passed = (
        unmatched_count == 0 and
        mean_iou >= thresholds.min_mean_iou and
        mean_score_delta <= thresholds.max_mean_score_delta and
        class_match_rate >= thresholds.min_class_match_rate
    )
    return FrameComparison(
        reference_index=reference.index,
        candidate_index=candidate.index,
        mean_iou=mean_iou,
        mean_score_delta=mean_score_delta,
        class_match_rate=class_match_rate,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        passed=passed,
    )


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def format_optional_float(value: Optional[float]) -> str:
    return f'{value:.4f}' if value is not None else 'inf'


def percentile(values: Sequence[float], pct: float) -> float:
    return float(np.percentile(np.array(values), pct)) if values else float('nan')


def summarize(
    comparisons: Sequence[FrameComparison],
    reference_frame_count: int,
    candidate_frame_count: int,
    unpaired_reference_frames: int,
    unpaired_candidate_frames: int,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    if not comparisons:
        return {
            'paired_frames': 0,
            'reference_frames': reference_frame_count,
            'candidate_frames': candidate_frame_count,
            'unpaired_reference_frames': unpaired_reference_frames,
            'unpaired_candidate_frames': unpaired_candidate_frames,
            'pass': False,
        }

    ious = [c.mean_iou for c in comparisons]
    finite_deltas = [c.mean_score_delta for c in comparisons if math.isfinite(c.mean_score_delta)]
    class_rates = [c.class_match_rate for c in comparisons]
    frame_pass_rate = float(np.mean([c.passed for c in comparisons]))
    unmatched_total = int(sum(c.unmatched_count for c in comparisons))
    unmatched_frames = int(sum(c.unmatched_count > 0 for c in comparisons))

    mean_score_delta = float(np.mean(finite_deltas)) if finite_deltas else float('inf')
    mean_iou = float(np.mean(ious))
    mean_class_match_rate = float(np.mean(class_rates))

    passed = (
        len(comparisons) >= thresholds.min_paired_frames and
        mean_iou >= thresholds.min_mean_iou and
        mean_score_delta <= thresholds.max_mean_score_delta and
        frame_pass_rate >= thresholds.min_frame_pass_rate and
        mean_class_match_rate >= thresholds.min_class_match_rate
    )

    return {
        'paired_frames': len(comparisons),
        'reference_frames': reference_frame_count,
        'candidate_frames': candidate_frame_count,
        'unpaired_reference_frames': unpaired_reference_frames,
        'unpaired_candidate_frames': unpaired_candidate_frames,
        'mean_iou': finite_or_none(mean_iou),
        'median_iou': finite_or_none(float(np.median(ious))),
        'p95_iou': finite_or_none(percentile(ious, 95)),
        'mean_score_delta': finite_or_none(mean_score_delta),
        'median_score_delta': finite_or_none(
            float(np.median(finite_deltas)) if finite_deltas else float('inf')),
        'p95_score_delta': finite_or_none(percentile(finite_deltas, 95)),
        'mean_class_match_rate': finite_or_none(mean_class_match_rate),
        'frame_pass_rate': frame_pass_rate,
        'unmatched_detections': unmatched_total,
        'unmatched_frames': unmatched_frames,
        'pass': passed,
    }


def print_summary(summary: Dict[str, Any], thresholds: Thresholds) -> None:
    print(f"Paired frames: {summary['paired_frames']}")
    print(f"Reference frames: {summary['reference_frames']}")
    print(f"Candidate frames: {summary['candidate_frames']}")
    print(f"Unpaired frames: reference={summary['unpaired_reference_frames']} "
          f"candidate={summary['unpaired_candidate_frames']}")
    if summary['paired_frames'] > 0:
        print(f"IoU   mean={summary['mean_iou']:.4f} "
              f"median={summary['median_iou']:.4f} p95={summary['p95_iou']:.4f}")
        score_mean = summary['mean_score_delta']
        score_median = summary['median_score_delta']
        score_p95 = summary['p95_score_delta']
        print('Score '
              f'mean={format_optional_float(score_mean)} '
              f'median={format_optional_float(score_median)} '
              f'p95={format_optional_float(score_p95)}')
        print(f"Class match rate: {summary['mean_class_match_rate']:.4f}")
        print(f"Unmatched detections: total={summary['unmatched_detections']} "
              f"frames={summary['unmatched_frames']}/{summary['paired_frames']}")
        print(f"Frames passing per-frame threshold: {summary['frame_pass_rate'] * 100:.1f}%")
    print('Thresholds: '
          f'min_paired_frames={thresholds.min_paired_frames}, '
          f'min_mean_iou={thresholds.min_mean_iou:.4f}, '
          f'max_mean_score_delta={thresholds.max_mean_score_delta:.4f}, '
          f'min_frame_pass_rate={thresholds.min_frame_pass_rate:.4f}, '
          f'min_class_match_rate={thresholds.min_class_match_rate:.4f}')
    print('PASS' if summary['pass'] else 'FAIL')


def write_report(path: str, report: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare two vision_msgs/msg/Detection2DArray output bags.')
    parser.add_argument('--reference-bag', required=True)
    parser.add_argument('--candidate-bag', required=True)
    parser.add_argument('--reference-topic', required=True)
    parser.add_argument('--candidate-topic', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--match-policy', choices=['stamp', 'index'], default='stamp')
    parser.add_argument('--storage-id', default='',
                        help='rosbag2 storage id override. Defaults to metadata.yaml value.')
    parser.add_argument('--min-mean-iou', type=float, default=0.80)
    parser.add_argument('--max-mean-score-delta', type=float, default=0.20)
    parser.add_argument('--min-frame-pass-rate', type=float, default=0.70)
    parser.add_argument('--min-paired-frames', type=int, default=20)
    parser.add_argument('--min-class-match-rate', type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(
        min_mean_iou=args.min_mean_iou,
        max_mean_score_delta=args.max_mean_score_delta,
        min_frame_pass_rate=args.min_frame_pass_rate,
        min_paired_frames=args.min_paired_frames,
        min_class_match_rate=args.min_class_match_rate,
    )

    try:
        reference_frames = read_detection_frames(
            args.reference_bag, args.reference_topic, args.storage_id)
        candidate_frames = read_detection_frames(
            args.candidate_bag, args.candidate_topic, args.storage_id)
        pairs, unpaired_reference, unpaired_candidate = pair_frames(
            reference_frames, candidate_frames, args.match_policy)
        comparisons = [compare_frame(ref, cand, thresholds) for ref, cand in pairs]
        summary = summarize(
            comparisons,
            len(reference_frames),
            len(candidate_frames),
            unpaired_reference,
            unpaired_candidate,
            thresholds,
        )
        report = {
            'status': 'PASS' if summary['pass'] else 'FAIL',
            'inputs': {
                'reference_bag': args.reference_bag,
                'candidate_bag': args.candidate_bag,
                'reference_topic': args.reference_topic,
                'candidate_topic': args.candidate_topic,
                'match_policy': args.match_policy,
            },
            'thresholds': {
                'min_mean_iou': thresholds.min_mean_iou,
                'max_mean_score_delta': thresholds.max_mean_score_delta,
                'min_frame_pass_rate': thresholds.min_frame_pass_rate,
                'min_paired_frames': thresholds.min_paired_frames,
                'min_class_match_rate': thresholds.min_class_match_rate,
            },
            'summary': summary,
        }
        print_summary(summary, thresholds)
        write_report(args.output_json, report)
        return 0 if summary['pass'] else 1
    except Exception as exc:  # noqa: BLE001 - CLI should convert all failures to a clear exit.
        report = {
            'status': 'ERROR',
            'error': str(exc),
            'inputs': {
                'reference_bag': args.reference_bag,
                'candidate_bag': args.candidate_bag,
                'reference_topic': args.reference_topic,
                'candidate_topic': args.candidate_topic,
                'match_policy': args.match_policy,
            },
        }
        print(f'ERROR: {exc}', file=sys.stderr)
        write_report(args.output_json, report)
        return 1


if __name__ == '__main__':
    sys.exit(main())
