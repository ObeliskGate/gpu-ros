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

"""Compare two Detection2DArray output bags."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
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
COMPARISON_SCHEMA = 'phase2b_detection_report_only_v2'


def comparator_source_sha256() -> str:
    """Return the source hash recorded with every comparison report."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DetectionFrame:
    """A Detection2DArray message plus bag/index metadata."""

    index: int
    stamp_ns: int
    bag_time_ns: int
    detections: Sequence[Detection2D]


@dataclass(frozen=True)
class FrameComparison:
    """Observed metrics for one paired frame, without a pass/fail judgment."""

    reference_index: int
    candidate_index: int
    reference_stamp_ns: int
    candidate_stamp_ns: int
    reference_bag_time_ns: int
    candidate_bag_time_ns: int
    mean_iou: Optional[float]
    mean_score_delta: Optional[float]
    class_match_rate: Optional[float]
    class_match_count: int
    class_match_sample_count: int
    matched_count: int
    unmatched_count: int
    unmatched_reference_count: int
    unmatched_candidate_count: int
    iou_values: Sequence[float]
    score_delta_values: Sequence[float]

    @property
    def class_mismatch_count(self) -> int:
        return self.class_match_sample_count - self.class_match_count


@dataclass(frozen=True)
class DetectionFilters:
    """Optional diagnostic filters; defaults preserve every detection."""

    min_score: Optional[float] = None
    max_detections_per_frame: int = 0


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


def resolve_detection_topic(
    topic_types: Dict[str, str],
    requested_topic: str,
    bag_path: str,
) -> str:
    if requested_topic and requested_topic != 'auto':
        normalized_topic = normalize_topic(requested_topic)
        if normalized_topic not in topic_types:
            available = ', '.join(sorted(topic_types))
            raise RuntimeError(
                f"Topic '{requested_topic}' not found in bag '{bag_path}'. "
                f"Available topics: {available}"
            )
        return normalized_topic

    detection_topics = sorted(
        topic for topic, topic_type in topic_types.items() if topic_type == DETECTION2D_ARRAY_TYPE
    )
    if len(detection_topics) == 1:
        return detection_topics[0]
    if not detection_topics:
        available = ', '.join(sorted(topic_types))
        raise RuntimeError(
            f"No '{DETECTION2D_ARRAY_TYPE}' topic found in bag '{bag_path}'. "
            f"Available topics: {available}"
        )
    raise RuntimeError(
        f"Multiple '{DETECTION2D_ARRAY_TYPE}' topics found in bag '{bag_path}': "
        f"{', '.join(detection_topics)}. Specify the topic explicitly."
    )


def stamp_key(msg: Detection2DArray) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def detection_score(det: Detection2D) -> Optional[float]:
    if not det.results:
        return None
    score = float(det.results[0].hypothesis.score)
    if not math.isfinite(score):
        raise ValueError(f'Detection score must be finite, got {score!r}.')
    return score


def detection_class_id(det: Detection2D) -> Optional[str]:
    return str(det.results[0].hypothesis.class_id) if det.results else None


def filter_detections(
    detections: Sequence[Detection2D],
    detection_filters: DetectionFilters,
) -> List[Detection2D]:
    if detection_filters.min_score is not None and not math.isfinite(detection_filters.min_score):
        raise ValueError('Minimum detection score filter must be finite.')
    if detection_filters.max_detections_per_frame < 0:
        raise ValueError('Maximum detections per frame must be nonnegative.')

    filtered: List[Tuple[Optional[float], Detection2D]] = []
    for detection in detections:
        # Validate before applying diagnostic filters so bad observations cannot
        # disappear behind a score threshold or top-K truncation.
        score = detection_score(detection)
        to_xyxy(detection)
        if detection_filters.min_score is None or (
            score is not None and score >= detection_filters.min_score
        ):
            filtered.append((score, detection))
    if detection_filters.max_detections_per_frame > 0:
        filtered.sort(
            key=lambda item: (item[0] is not None, item[0] if item[0] is not None else 0.0),
            reverse=True,
        )
        filtered = filtered[: detection_filters.max_detections_per_frame]
    return [detection for _, detection in filtered]


def to_xyxy(det: Detection2D) -> Tuple[float, float, float, float]:
    cx = float(det.bbox.center.position.x)
    cy = float(det.bbox.center.position.y)
    width = float(det.bbox.size_x)
    height = float(det.bbox.size_y)
    values = (cx, cy, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'Detection bounding box values must be finite, got {values!r}.')
    if width < 0.0 or height < 0.0:
        raise ValueError(
            f'Detection bounding box sizes must be nonnegative, got ({width!r}, {height!r}).'
        )
    area = width * height
    if not math.isfinite(area):
        raise ValueError(f'Detection bounding box area overflowed: {area!r}.')
    box = (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'Detection bounding box coordinates overflowed: {box!r}.')
    return box


def iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    if not all(math.isfinite(value) for value in (*box_a, *box_b)):
        raise ValueError('IoU bounding box coordinates must be finite.')
    if box_a[2] < box_a[0] or box_a[3] < box_a[1]:
        raise ValueError(f'Invalid reference bounding box: {box_a!r}.')
    if box_b[2] < box_b[0] or box_b[3] < box_b[1]:
        raise ValueError(f'Invalid candidate bounding box: {box_b!r}.')
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if not all(math.isfinite(value) for value in (intersection, area_a, area_b, union)):
        raise ValueError('IoU calculation overflowed for bounding box coordinates.')
    if union < 0.0:
        raise ValueError(f'Invalid negative bounding box union: {union!r}.')
    return intersection / union if union > 0.0 else 0.0


def unpaired_detection_counts(
    reference: Sequence[DetectionFrame],
    candidate: Sequence[DetectionFrame],
    pairs: Sequence[Tuple[DetectionFrame, DetectionFrame]],
) -> Tuple[int, int]:
    """Count detections belonging to frames that were not paired."""
    paired_reference_indices = {frame.index for frame, _ in pairs}
    paired_candidate_indices = {frame.index for _, frame in pairs}
    return (
        sum(
            len(frame.detections)
            for frame in reference
            if frame.index not in paired_reference_indices
        ),
        sum(
            len(frame.detections)
            for frame in candidate
            if frame.index not in paired_candidate_indices
        ),
    )


def read_detection_frames(
    bag_path: str,
    topic: str,
    detection_filters: DetectionFilters,
    storage_id: str = '',
) -> Tuple[List[DetectionFrame], str]:
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

    topic_types = {normalize_topic(t.name): t.type for t in reader.get_all_topics_and_types()}
    normalized_topic = resolve_detection_topic(topic_types, topic, bag_path)
    if topic_types[normalized_topic] != DETECTION2D_ARRAY_TYPE:
        raise RuntimeError(
            f"Topic '{normalized_topic}' has type '{topic_types[normalized_topic]}', "
            f"expected '{DETECTION2D_ARRAY_TYPE}'."
        )

    frames: List[DetectionFrame] = []
    while reader.has_next():
        topic_name, data, bag_time_ns = reader.read_next()
        if not topic_matches(topic_name, normalized_topic):
            continue
        msg = deserialize_message(data, Detection2DArray)
        frames.append(
            DetectionFrame(
                index=len(frames),
                stamp_ns=stamp_key(msg),
                bag_time_ns=int(bag_time_ns),
                detections=filter_detections(msg.detections, detection_filters),
            )
        )
    return frames, normalized_topic


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
    class_aware: bool = False,
) -> Tuple[List[float], List[float], int, int, int, int]:
    reference_scores = [detection_score(detection) for detection in reference_detections]
    candidate_scores = [detection_score(detection) for detection in candidate_detections]
    reference_classes = [detection_class_id(detection) for detection in reference_detections]
    candidate_classes = [detection_class_id(detection) for detection in candidate_detections]
    reference_boxes = [to_xyxy(detection) for detection in reference_detections]
    candidate_boxes = [to_xyxy(detection) for detection in candidate_detections]

    if not reference_detections or not candidate_detections:
        return [], [], len(reference_detections), len(candidate_detections), 0, 0

    candidate_pairs = []
    for ref_index in range(len(reference_detections)):
        for cand_index in range(len(candidate_detections)):
            ref_class = reference_classes[ref_index]
            cand_class = candidate_classes[cand_index]
            if class_aware and ref_class != cand_class:
                continue
            candidate_pairs.append(
                (
                    iou(reference_boxes[ref_index], candidate_boxes[cand_index]),
                    ref_index,
                    cand_index,
                )
            )
    # IoU is the primary order; source indices make ties deterministic.
    candidate_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    used_reference = set()
    used_candidate = set()
    ious = []
    score_deltas = []
    class_matches = 0
    class_match_samples = 0

    for pair_iou, ref_index, cand_index in candidate_pairs:
        if pair_iou <= 0.0:
            break
        if ref_index in used_reference or cand_index in used_candidate:
            continue
        used_reference.add(ref_index)
        used_candidate.add(cand_index)
        ious.append(pair_iou)

        ref_score = reference_scores[ref_index]
        cand_score = candidate_scores[cand_index]
        if ref_score is not None and cand_score is not None:
            score_deltas.append(abs(ref_score - cand_score))

        ref_class = reference_classes[ref_index]
        cand_class = candidate_classes[cand_index]
        if ref_class is not None and cand_class is not None:
            class_match_samples += 1
            if ref_class == cand_class:
                class_matches += 1

    unmatched_reference_count = len(reference_detections) - len(used_reference)
    unmatched_candidate_count = len(candidate_detections) - len(used_candidate)
    return (
        ious,
        score_deltas,
        unmatched_reference_count,
        unmatched_candidate_count,
        class_matches,
        class_match_samples,
    )


def compare_frame(
    reference: DetectionFrame,
    candidate: DetectionFrame,
    class_aware_matching: bool = False,
) -> FrameComparison:
    (
        ious,
        score_deltas,
        unmatched_reference_count,
        unmatched_candidate_count,
        class_matches,
        class_match_samples,
    ) = match_detections(
        reference.detections,
        candidate.detections,
        class_aware=class_aware_matching,
    )
    matched_count = len(ious)
    return FrameComparison(
        reference_index=reference.index,
        candidate_index=candidate.index,
        reference_stamp_ns=reference.stamp_ns,
        candidate_stamp_ns=candidate.stamp_ns,
        reference_bag_time_ns=reference.bag_time_ns,
        candidate_bag_time_ns=candidate.bag_time_ns,
        mean_iou=float(np.mean(ious)) if ious else None,
        mean_score_delta=float(np.mean(score_deltas)) if score_deltas else None,
        class_match_rate=(class_matches / class_match_samples if class_match_samples else None),
        class_match_count=class_matches,
        class_match_sample_count=class_match_samples,
        matched_count=matched_count,
        unmatched_count=unmatched_reference_count + unmatched_candidate_count,
        unmatched_reference_count=unmatched_reference_count,
        unmatched_candidate_count=unmatched_candidate_count,
        iou_values=tuple(ious),
        score_delta_values=tuple(score_deltas),
    )


def finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f'Comparison produced a non-finite metric: {value!r}.')
    return float(value)


def format_optional_float(value: Optional[float]) -> str:
    return f'{value:.4f}' if value is not None else 'n/a'


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Cannot summarize non-finite comparison samples.')
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def comparison_to_dict(comparison: FrameComparison) -> Dict[str, Any]:
    return {
        'reference_index': comparison.reference_index,
        'candidate_index': comparison.candidate_index,
        'reference_stamp_ns': comparison.reference_stamp_ns,
        'candidate_stamp_ns': comparison.candidate_stamp_ns,
        'reference_bag_time_ns': comparison.reference_bag_time_ns,
        'candidate_bag_time_ns': comparison.candidate_bag_time_ns,
        'mean_iou': finite_or_none(comparison.mean_iou),
        'mean_score_delta': finite_or_none(comparison.mean_score_delta),
        'class_match_rate': finite_or_none(comparison.class_match_rate),
        'class_match_count': comparison.class_match_count,
        'class_match_sample_count': comparison.class_match_sample_count,
        'class_mismatch_count': comparison.class_mismatch_count,
        'matched_count': comparison.matched_count,
        'score_delta_sample_count': len(comparison.score_delta_values),
        'unmatched_count': comparison.unmatched_count,
        'unmatched_reference_count': comparison.unmatched_reference_count,
        'unmatched_candidate_count': comparison.unmatched_candidate_count,
        'iou_values': [finite_or_none(value) for value in comparison.iou_values],
        'score_delta_values': [finite_or_none(value) for value in comparison.score_delta_values],
    }


def distribution(values: Sequence[float]) -> Dict[str, Any]:
    """Return statistics over actual finite samples, with nulls for no samples."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Cannot summarize non-finite comparison samples.')
    if not values:
        return {
            'count': 0,
            'mean': None,
            'min': None,
            'p05': None,
            'median': None,
            'p95': None,
            'max': None,
        }
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'min': float(np.min(values)),
        'p05': percentile(values, 5),
        'median': float(np.median(values)),
        'p95': percentile(values, 95),
        'max': float(np.max(values)),
    }


def comparison_to_unpaired_frame(frame: DetectionFrame) -> Dict[str, int]:
    return {
        'index': frame.index,
        'stamp_ns': frame.stamp_ns,
        'bag_time_ns': frame.bag_time_ns,
        'detection_count': len(frame.detections),
    }


def worst_frame_details(
    comparisons: Sequence[FrameComparison],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    def sort_key(
        comparison: FrameComparison,
    ) -> Tuple[int, int, float, float, int, int]:
        # Rank observed differences only; absent metrics are not encoded as
        # synthetic values in the report and sort behind real samples.
        score_delta = (
            comparison.mean_score_delta if comparison.mean_score_delta is not None else -1.0
        )
        mean_iou = comparison.mean_iou if comparison.mean_iou is not None else 1.0
        return (
            -comparison.unmatched_count,
            -comparison.class_mismatch_count,
            -score_delta,
            mean_iou,
            comparison.reference_index,
            comparison.candidate_index,
        )

    return [
        comparison_to_dict(comparison) for comparison in sorted(comparisons, key=sort_key)[:limit]
    ]


def summarize(
    comparisons: Sequence[FrameComparison],
    reference_frame_count: int,
    candidate_frame_count: int,
    unpaired_reference_frames: int,
    unpaired_candidate_frames: int,
    unpaired_reference_detections: int = 0,
    unpaired_candidate_detections: int = 0,
) -> Dict[str, Any]:
    pair_ious = [value for comparison in comparisons for value in comparison.iou_values]
    pair_score_deltas = [
        value for comparison in comparisons for value in comparison.score_delta_values
    ]
    frame_class_rates = [
        comparison.class_match_rate
        for comparison in comparisons
        if comparison.class_match_rate is not None
    ]
    iou_stats = distribution(pair_ious)
    score_delta_stats = distribution(pair_score_deltas)
    class_match_rate_stats = distribution(frame_class_rates)

    unmatched_reference_total = int(
        unpaired_reference_detections + sum(c.unmatched_reference_count for c in comparisons)
    )
    unmatched_candidate_total = int(
        unpaired_candidate_detections + sum(c.unmatched_candidate_count for c in comparisons)
    )
    matched_total = int(sum(c.matched_count for c in comparisons))
    class_match_total = int(sum(c.class_match_count for c in comparisons))
    class_match_sample_total = int(sum(c.class_match_sample_count for c in comparisons))
    unmatched_frames = int(
        unpaired_reference_frames
        + unpaired_candidate_frames
        + sum(c.unmatched_count > 0 for c in comparisons)
    )
    paired_count = len(comparisons)

    return {
        'paired_frames': paired_count,
        'reference_frames': reference_frame_count,
        'candidate_frames': candidate_frame_count,
        'unpaired_reference_frames': unpaired_reference_frames,
        'unpaired_candidate_frames': unpaired_candidate_frames,
        'frame_coverage': {
            'paired_frames': paired_count,
            'reference_frame_count': reference_frame_count,
            'candidate_frame_count': candidate_frame_count,
            'unpaired_reference_frame_count': unpaired_reference_frames,
            'unpaired_candidate_frame_count': unpaired_candidate_frames,
            'reference_paired_rate': (
                paired_count / reference_frame_count if reference_frame_count else None
            ),
            'candidate_paired_rate': (
                paired_count / candidate_frame_count if candidate_frame_count else None
            ),
        },
        'mean_iou': iou_stats['mean'],
        'min_iou': iou_stats['min'],
        'p05_iou': iou_stats['p05'],
        'median_iou': iou_stats['median'],
        'p95_iou': iou_stats['p95'],
        'max_iou': iou_stats['max'],
        'mean_score_delta': score_delta_stats['mean'],
        'p05_score_delta': score_delta_stats['p05'],
        'median_score_delta': score_delta_stats['median'],
        'p95_score_delta': score_delta_stats['p95'],
        'max_score_delta': score_delta_stats['max'],
        'score_delta_sample_count': len(pair_score_deltas),
        'class_match_count': class_match_total,
        'class_match_sample_count': class_match_sample_total,
        'class_match_rate_denominator': class_match_sample_total,
        'class_match_rate': (
            class_match_total / class_match_sample_total if class_match_sample_total else None
        ),
        'unpaired_reference_detections': unpaired_reference_detections,
        'unpaired_candidate_detections': unpaired_candidate_detections,
        'unmatched_detections': unmatched_reference_total + unmatched_candidate_total,
        'matched_detections': matched_total,
        'unmatched_reference_detections': unmatched_reference_total,
        'unmatched_candidate_detections': unmatched_candidate_total,
        'unmatched_frames': unmatched_frames,
        'aggregate_distributions': {
            'iou': iou_stats,
            'score_delta': score_delta_stats,
            'class_match_rate_per_frame': class_match_rate_stats,
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"Paired frames: {summary['paired_frames']}")
    print(f"Reference frames: {summary['reference_frames']}")
    print(f"Candidate frames: {summary['candidate_frames']}")
    print(
        f"Unpaired frames: reference={summary['unpaired_reference_frames']} "
        f"candidate={summary['unpaired_candidate_frames']}"
    )
    if summary['aggregate_distributions']['iou']['count']:
        print(
            f"IoU   mean={summary['mean_iou']:.4f} min={summary['min_iou']:.4f} "
            f"p05={summary['p05_iou']:.4f} median={summary['median_iou']:.4f}"
        )
        print(
            'Score '
            f"mean={format_optional_float(summary['mean_score_delta'])} "
            f"median={format_optional_float(summary['median_score_delta'])} "
            f"p95={format_optional_float(summary['p95_score_delta'])} "
            f"max={format_optional_float(summary['max_score_delta'])}"
        )
    else:
        print('Matched detections: 0; IoU and score metrics are not available.')
    denominator = summary['class_match_rate_denominator']
    if denominator:
        print(
            f"Class match rate: {summary['class_match_rate']:.4f} "
            f"({summary['class_match_count']}/{denominator} matched detections)"
        )
    else:
        print('Class match rate: n/a (0/0 matched detections)')
    print(
        f"Unmatched detections: total={summary['unmatched_detections']} "
        f"frames={summary['unmatched_frames']}"
    )
    print('REPORT_ONLY')


def write_report(path: str, report: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare two vision_msgs/msg/Detection2DArray output bags.'
    )
    parser.add_argument('--reference-bag', required=True)
    parser.add_argument('--candidate-bag', required=True)
    parser.add_argument(
        '--reference-topic',
        default='auto',
        help="Detection2DArray topic or 'auto' to use the only one in the bag.",
    )
    parser.add_argument(
        '--candidate-topic',
        default='auto',
        help="Detection2DArray topic or 'auto' to use the only one in the bag.",
    )
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--match-policy', choices=['stamp', 'index'], default='stamp')
    parser.add_argument(
        '--storage-id',
        default='',
        help='rosbag2 storage id override. Defaults to metadata.yaml value.',
    )
    parser.add_argument(
        '--class-aware-matching',
        action='store_true',
        help='Only consider same-class detection pairs during IoU matching.',
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=None,
        help='Diagnostic filter: drop detections below this confidence score.',
    )
    parser.add_argument(
        '--max-detections-per-frame',
        type=int,
        default=0,
        help='Diagnostic filter: keep top K detections by score; 0 keeps all.',
    )
    parser.add_argument(
        '--max-frame-details',
        type=int,
        default=20,
        help='Maximum number of raw-difference-ranked frame details to write.',
    )
    return parser.parse_args()


def comparison_method(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'frame_pairing': {
            'policy': args.match_policy,
            'description': (
                'exact source header stamp; duplicate stamps paired FIFO in bag order'
                if args.match_policy == 'stamp'
                else 'bag-order index pairing; source header stamps are not compared'
            ),
        },
        'detection_pairing': {
            'algorithm': 'descending-IoU deterministic greedy one-to-one; only IoU > 0',
            'tie_breaker': 'reference detection index, then candidate detection index',
            'class_aware': args.class_aware_matching,
            'missing_class_metadata': (
                'excluded from class-match samples; with class-aware matching, '
                'pairs only with another detection missing class metadata'
            ),
        },
        'box_normalization': 'Detection2D center and size converted to x1/y1/x2/y2',
        'confidence_filter': (
            f'score >= {args.min_score}' if args.min_score is not None else 'none'
        ),
        'missing_score_metadata': {
            'without_min_score_filter': (
                'retained for geometric matching; excluded from score-delta samples'
            ),
            'with_min_score_filter': 'excluded because a numeric score is unavailable',
            'with_top_k': 'ranked after detections with numeric scores',
        },
        'top_k_truncation': (
            f'top {args.max_detections_per_frame} detections by score'
            if args.max_detections_per_frame > 0
            else 'none'
        ),
        'pair_metrics': (
            'IoU per geometric match; score delta and class equality only when both '
            'detections provide hypothesis results'
        ),
        'worst_frame_details': {
            'max_entries': max(0, args.max_frame_details),
            'ordering': (
                'descending unmatched detections; descending class mismatches; '
                'descending score delta; ascending IoU; source frame indices'
            ),
        },
        'aggregate_status': 'REPORT_ONLY; numeric observations do not determine exit status',
    }


def input_identity(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        'reference_bag': args.reference_bag,
        'candidate_bag': args.candidate_bag,
        'reference_topic': args.reference_topic,
        'candidate_topic': args.candidate_topic,
        'match_policy': args.match_policy,
        'storage_id_override': args.storage_id or None,
    }


def main() -> int:
    args = parse_args()
    detection_filters = DetectionFilters(
        min_score=args.min_score,
        max_detections_per_frame=args.max_detections_per_frame,
    )
    method: Optional[Dict[str, Any]] = None
    try:
        # Validate diagnostic options even when both bags contain no frames.
        filter_detections([], detection_filters)
        method = comparison_method(args)
        reference_storage_id = resolve_storage_id(args.reference_bag, args.storage_id)
        candidate_storage_id = resolve_storage_id(args.candidate_bag, args.storage_id)
        reference_frames, resolved_reference_topic = read_detection_frames(
            args.reference_bag, args.reference_topic, detection_filters, args.storage_id
        )
        candidate_frames, resolved_candidate_topic = read_detection_frames(
            args.candidate_bag, args.candidate_topic, detection_filters, args.storage_id
        )
        pairs, unpaired_reference, unpaired_candidate = pair_frames(
            reference_frames, candidate_frames, args.match_policy
        )
        unpaired_reference_detections, unpaired_candidate_detections = unpaired_detection_counts(
            reference_frames, candidate_frames, pairs
        )
        comparisons = [
            compare_frame(ref, cand, class_aware_matching=args.class_aware_matching)
            for ref, cand in pairs
        ]
        summary = summarize(
            comparisons,
            len(reference_frames),
            len(candidate_frames),
            unpaired_reference,
            unpaired_candidate,
            unpaired_reference_detections=unpaired_reference_detections,
            unpaired_candidate_detections=unpaired_candidate_detections,
        )
        paired_reference_indices = {reference.index for reference, _ in pairs}
        paired_candidate_indices = {candidate.index for _, candidate in pairs}
        inputs = input_identity(args)
        inputs['reference_storage_id'] = reference_storage_id or None
        inputs['resolved_reference_topic'] = resolved_reference_topic
        inputs['resolved_candidate_topic'] = resolved_candidate_topic
        inputs['candidate_storage_id'] = candidate_storage_id or None
        report = {
            'status': 'REPORT_ONLY',
            'comparison_schema': COMPARISON_SCHEMA,
            'comparison_method': method,
            'comparator_source_sha256': comparator_source_sha256(),
            'cli_arguments': sys.argv[1:],
            'inputs': inputs,
            'filters': {
                'min_score': detection_filters.min_score,
                'max_detections_per_frame': detection_filters.max_detections_per_frame,
            },
            'summary': summary,
            'per_frame': [comparison_to_dict(comparison) for comparison in comparisons],
            'unpaired_frames': {
                'reference': [
                    comparison_to_unpaired_frame(frame)
                    for frame in reference_frames
                    if frame.index not in paired_reference_indices
                ],
                'candidate': [
                    comparison_to_unpaired_frame(frame)
                    for frame in candidate_frames
                    if frame.index not in paired_candidate_indices
                ],
            },
            'worst_frames': worst_frame_details(comparisons, args.max_frame_details),
        }
    except Exception as exc:  # noqa: BLE001 - CLI converts input failures to a clear report.
        report = {
            'status': 'ERROR',
            'error': str(exc),
            'comparison_schema': COMPARISON_SCHEMA,
            'comparator_source_sha256': comparator_source_sha256(),
            'cli_arguments': sys.argv[1:],
            'inputs': input_identity(args),
        }
        if method is not None:
            report['comparison_method'] = method
            report['filters'] = {
                'min_score': detection_filters.min_score,
                'max_detections_per_frame': detection_filters.max_detections_per_frame,
            }
        print(f'ERROR: {exc}', file=sys.stderr)
        try:
            write_report(args.output_json, report)
        except Exception as report_exc:  # noqa: BLE001 - report writing is itself an input error.
            print(f'ERROR: failed to write error report: {report_exc}', file=sys.stderr)
        return 1

    try:
        write_report(args.output_json, report)
    except Exception as exc:  # noqa: BLE001 - a report write failure is a CLI failure.
        print(f'ERROR: failed to write report: {exc}', file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
