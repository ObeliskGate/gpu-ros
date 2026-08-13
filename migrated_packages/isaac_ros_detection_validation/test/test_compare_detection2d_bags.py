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

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from rclpy.serialization import serialize_message
import rosbag2_py
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] /
    'scripts' /
    'compare_detection2d_bags.py'
)
SPEC = importlib.util.spec_from_file_location('compare_detection2d_bags', SCRIPT_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def make_detection(cx, cy, width, height, score=0.9, class_id='0'):
    det = Detection2D()
    det.bbox.center.position.x = float(cx)
    det.bbox.center.position.y = float(cy)
    det.bbox.size_x = float(width)
    det.bbox.size_y = float(height)
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = str(class_id)
    result.hypothesis.score = float(score)
    det.results.append(result)
    return det


def make_msg(stamp_ns, detections):
    msg = Detection2DArray()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.detections = detections
    return msg


def frame(index, stamp_ns, detections):
    return compare.DetectionFrame(
        index=index,
        stamp_ns=stamp_ns,
        bag_time_ns=stamp_ns,
        detections=detections,
    )


def thresholds():
    return compare.Thresholds(
        min_mean_iou=0.80,
        max_mean_score_delta=0.20,
        min_frame_pass_rate=0.70,
        min_paired_frames=1,
        min_class_match_rate=1.0,
    )


def detection_filters(min_score=0.0, max_detections_per_frame=0):
    return compare.DetectionFilters(
        min_score=min_score,
        max_detections_per_frame=max_detections_per_frame,
    )


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert compare.iou(box, box) == 1.0


def test_filter_detections_applies_score_and_top_k():
    detections = [
        make_detection(0, 0, 1, 1, score=0.30),
        make_detection(0, 0, 1, 1, score=0.95),
        make_detection(0, 0, 1, 1, score=0.80),
    ]

    filtered = compare.filter_detections(
        detections,
        detection_filters(min_score=0.50, max_detections_per_frame=1),
    )

    assert len(filtered) == 1
    assert compare.detection_score(filtered[0]) == 0.95


def test_resolve_detection_topic_auto_selects_single_detection_topic():
    topic = compare.resolve_detection_topic(
        {
            '/image': 'sensor_msgs/msg/Image',
            '/ns/detections_output': compare.DETECTION2D_ARRAY_TYPE,
        },
        requested_topic='auto',
        bag_path='bag',
    )

    assert topic == '/ns/detections_output'


def test_resolve_detection_topic_auto_rejects_multiple_detection_topics():
    try:
        compare.resolve_detection_topic(
            {
                '/a/detections_output': compare.DETECTION2D_ARRAY_TYPE,
                '/b/detections_output': compare.DETECTION2D_ARRAY_TYPE,
            },
            requested_topic='auto',
            bag_path='bag',
        )
    except RuntimeError as exc:
        assert 'Multiple' in str(exc)
    else:
        raise AssertionError('Expected multiple detection topics to fail')


def test_empty_frames_pass():
    result = compare.compare_frame(frame(0, 1, []), frame(0, 1, []), thresholds())
    assert result.passed
    assert result.mean_iou == 1.0
    assert result.unmatched_count == 0


def test_one_sided_missing_detection_fails():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4)])
    candidate = frame(0, 1, [])
    result = compare.compare_frame(reference, candidate, thresholds())
    assert not result.passed
    assert result.unmatched_count == 1
    assert result.unmatched_reference_count == 1
    assert result.unmatched_candidate_count == 0


def test_matching_scores_and_classes_pass():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4, score=0.90, class_id='cup')])
    candidate = frame(0, 1, [make_detection(10.2, 10.1, 4, 4, score=0.86, class_id='cup')])
    result = compare.compare_frame(reference, candidate, thresholds())
    assert result.passed
    assert result.mean_iou > 0.80
    assert result.mean_score_delta < 0.20
    assert result.class_match_rate == 1.0


def test_class_mismatch_fails():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='cup')])
    candidate = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='box')])
    result = compare.compare_frame(reference, candidate, thresholds())
    assert not result.passed
    assert result.class_match_rate == 0.0


def test_pair_frames_by_index():
    reference = [frame(0, 10, []), frame(1, 20, [])]
    candidate = [frame(0, 99, [])]
    pairs, unpaired_reference, unpaired_candidate = compare.pair_frames(
        reference, candidate, 'index')
    assert len(pairs) == 1
    assert unpaired_reference == 1
    assert unpaired_candidate == 0


def test_pair_frames_by_stamp():
    reference = [frame(0, 10, []), frame(1, 20, [])]
    candidate = [frame(0, 20, []), frame(1, 30, [])]
    pairs, unpaired_reference, unpaired_candidate = compare.pair_frames(
        reference, candidate, 'stamp')
    assert [(ref.index, cand.index) for ref, cand in pairs] == [(1, 0)]
    assert unpaired_reference == 1
    assert unpaired_candidate == 1


def test_pair_frames_by_stamp_preserves_fifo_for_duplicate_stamps():
    reference = [frame(0, 10, []), frame(1, 10, [])]
    candidate = [frame(0, 10, []), frame(1, 10, [])]
    pairs, unpaired_reference, unpaired_candidate = compare.pair_frames(
        reference, candidate, 'stamp')
    assert [(ref.index, cand.index) for ref, cand in pairs] == [(0, 0), (1, 1)]
    assert unpaired_reference == 0
    assert unpaired_candidate == 0


def test_unpaired_frames_count_against_overall_frame_pass_rate():
    comparison = compare.compare_frame(
        frame(0, 1, [make_detection(10, 10, 4, 4)]),
        frame(0, 1, [make_detection(10, 10, 4, 4)]),
        thresholds(),
    )

    summary = compare.summarize(
        [comparison],
        reference_frame_count=2,
        candidate_frame_count=1,
        unpaired_reference_frames=1,
        unpaired_candidate_frames=0,
        thresholds=thresholds(),
    )

    assert summary['paired_frame_pass_rate'] == 1.0
    assert summary['frame_pass_rate'] == 0.5
    assert not summary['pass']


def test_unpaired_frames_can_be_ignored_for_benchmark_sweeps():
    comparison = compare.compare_frame(
        frame(0, 1, [make_detection(10, 10, 4, 4)]),
        frame(0, 1, [make_detection(10, 10, 4, 4)]),
        thresholds(),
    )

    summary = compare.summarize(
        [comparison],
        reference_frame_count=2,
        candidate_frame_count=1,
        unpaired_reference_frames=1,
        unpaired_candidate_frames=0,
        thresholds=thresholds(),
        ignore_unpaired_frames=True,
    )

    assert summary['total_evaluated_frames'] == 1
    assert summary['paired_frame_pass_rate'] == 1.0
    assert summary['frame_pass_rate'] == 1.0
    assert summary['ignore_unpaired_frames']
    assert summary['pass']


def test_worst_frame_details_prioritizes_failures():
    passing = compare.FrameComparison(
        reference_index=0,
        candidate_index=0,
        mean_iou=1.0,
        mean_score_delta=0.0,
        class_match_rate=1.0,
        matched_count=1,
        unmatched_count=0,
        passed=True,
    )
    failing = compare.FrameComparison(
        reference_index=1,
        candidate_index=1,
        mean_iou=0.0,
        mean_score_delta=float('inf'),
        class_match_rate=0.0,
        matched_count=0,
        unmatched_count=1,
        passed=False,
    )

    details = compare.worst_frame_details([passing, failing], limit=1)

    assert details[0]['reference_index'] == 1
    assert details[0]['mean_score_delta'] is None


def write_detection_bag(path, topic, messages):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    try:
        topic_metadata = rosbag2_py.TopicMetadata(
            id=0,
            name=topic,
            type=compare.DETECTION2D_ARRAY_TYPE,
            serialization_format='cdr',
            offered_qos_profiles=[],
        )
    except TypeError:
        topic_metadata = rosbag2_py.TopicMetadata()
        topic_metadata.name = topic
        topic_metadata.type = compare.DETECTION2D_ARRAY_TYPE
        topic_metadata.serialization_format = 'cdr'
    writer.create_topic(topic_metadata)

    for index, msg in enumerate(messages):
        writer.write(topic, serialize_message(msg), index + 1)


def test_cli_compares_detection_bags(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'

    write_detection_bag(
        reference_bag,
        topic,
        [make_msg(10, [make_detection(10, 10, 4, 4, score=0.90, class_id='cup')])],
    )
    write_detection_bag(
        candidate_bag,
        topic,
        [make_msg(99, [make_detection(10.1, 10.1, 4, 4, score=0.88, class_id='cup')])],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--reference-bag', str(reference_bag),
            '--candidate-bag', str(candidate_bag),
            '--match-policy', 'index',
            '--storage-id', 'sqlite3',
            '--min-paired-frames', '1',
            '--output-json', str(output_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output_json.read_text(encoding='utf-8'))
    assert report['status'] == 'PASS'
    assert report['summary']['paired_frames'] == 1
    assert report['summary']['total_evaluated_frames'] == 1
    assert report['filters']['min_score'] == 0.0
    assert len(report['worst_frames']) == 1


def test_cli_report_only_uses_fixed_method_and_has_full_frame_metrics(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    stamp_ns = 10_000_000_000

    message = make_msg(
        stamp_ns,
        [make_detection(10, 10, 4, 4, score=0.90, class_id='cup')],
    )
    write_detection_bag(reference_bag, topic, [message])
    write_detection_bag(candidate_bag, topic, [message])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--reference-bag', str(reference_bag),
            '--candidate-bag', str(candidate_bag),
            '--match-policy', 'stamp',
            '--storage-id', 'sqlite3',
            '--report-only',
            '--output-json', str(output_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output_json.read_text(encoding='utf-8'))
    assert report['status'] == 'REPORT_ONLY'
    assert report['comparison_schema'] == compare.COMPARISON_SCHEMA
    assert report['comparison_method']['confidence_filter'] == 'none'
    assert report['comparison_method']['top_k_truncation'] == 'none'
    assert len(report['per_frame']) == 1
    assert 'pass' not in report['summary']
    assert 'aggregate_pass' not in report['summary']
    assert report['summary']['unmatched_reference_detections'] == 0
    assert report['summary']['unmatched_candidate_detections'] == 0
    assert report['summary']['aggregate_distributions']['iou']['count'] == 1
    assert len(report['comparator_source_sha256']) == 64


def test_report_only_rejects_index_matching():
    values = dict(compare.REPORT_ONLY_DEFAULTS)
    values.update(report_only=True, match_policy='index')
    args = argparse.Namespace(**values)

    try:
        compare.validate_report_only_args(args)
    except ValueError as exc:
        assert 'match_policy' in str(exc)
    else:
        raise AssertionError('Expected report-only index matching to fail')
