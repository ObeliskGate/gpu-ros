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

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from rclpy.serialization import serialize_message
import rosbag2_py
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_detection2d_bags.py'
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


def make_detection_without_results(cx, cy, width, height):
    detection = make_detection(cx, cy, width, height)
    detection.results.clear()
    return detection


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


def detection_filters(min_score=None, max_detections_per_frame=0):
    return compare.DetectionFilters(
        min_score=min_score,
        max_detections_per_frame=max_detections_per_frame,
    )


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert compare.iou(box, box) == 1.0


def test_diagnostic_filters_apply_score_then_top_k():
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


def test_nonfinite_detection_score_is_rejected_before_score_filter():
    detection = make_detection(0, 0, 1, 1, score=float('nan'))

    with pytest.raises(ValueError, match='score must be finite'):
        compare.filter_detections([detection], detection_filters(min_score=0.5))


def test_nonfinite_bbox_is_rejected_before_score_filter():
    detection = make_detection(0, 0, 1, 1, score=0.1)
    detection.bbox.center.position.x = float('inf')

    with pytest.raises(ValueError, match='bounding box values must be finite'):
        compare.filter_detections([detection], detection_filters(min_score=0.5))


def test_negative_bbox_size_is_rejected_before_top_k():
    detection = make_detection(0, 0, -1, 1, score=0.1)

    with pytest.raises(ValueError, match='sizes must be nonnegative'):
        compare.filter_detections([detection], detection_filters(max_detections_per_frame=1))


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
    with pytest.raises(RuntimeError, match='Multiple'):
        compare.resolve_detection_topic(
            {
                '/a/detections_output': compare.DETECTION2D_ARRAY_TYPE,
                '/b/detections_output': compare.DETECTION2D_ARRAY_TYPE,
            },
            requested_topic='auto',
            bag_path='bag',
        )


def test_empty_detection_frames_have_no_comparable_metric_samples():
    comparison = compare.compare_frame(frame(0, 1, []), frame(0, 1, []))

    assert comparison.matched_count == 0
    assert comparison.unmatched_count == 0
    assert comparison.mean_iou is None
    assert comparison.mean_score_delta is None
    assert comparison.class_match_rate is None

    summary = compare.summarize([comparison], 1, 1, 0, 0)
    assert summary['paired_frames'] == 1
    assert summary['matched_detections'] == 0
    assert summary['class_match_rate_denominator'] == 0
    assert summary['class_match_rate'] is None
    assert summary['aggregate_distributions']['iou']['count'] == 0
    assert summary['aggregate_distributions']['iou']['mean'] is None
    assert summary['aggregate_distributions']['score_delta']['count'] == 0
    assert summary['aggregate_distributions']['class_match_rate_per_frame']['count'] == 0


def test_one_sided_empty_detection_frame_reports_unmatched_coverage():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4)])
    candidate = frame(0, 1, [])
    comparison = compare.compare_frame(reference, candidate)

    assert comparison.mean_iou is None
    assert comparison.mean_score_delta is None
    assert comparison.class_match_rate is None
    assert comparison.matched_count == 0
    assert comparison.unmatched_reference_count == 1
    assert comparison.unmatched_candidate_count == 0

    summary = compare.summarize([comparison], 1, 1, 0, 0)
    assert summary['unmatched_reference_detections'] == 1
    assert summary['unmatched_candidate_detections'] == 0
    assert summary['class_match_rate'] is None


def test_detection_differences_have_weighted_class_metrics():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4, score=0.90, class_id='cup')])
    candidate = frame(0, 1, [make_detection(10.2, 10.1, 4, 4, score=0.10, class_id='box')])
    comparison = compare.compare_frame(reference, candidate)

    assert comparison.mean_iou > 0.80
    assert comparison.mean_score_delta == pytest.approx(0.8)
    assert comparison.class_match_sample_count == 1
    assert len(comparison.score_delta_values) == 1
    assert comparison.class_match_count == 0
    assert comparison.class_match_rate == 0.0
    assert comparison.matched_count == 1
    assert comparison.unmatched_count == 0

    summary = compare.summarize([comparison], 1, 1, 0, 0)
    assert summary['class_match_count'] == 0
    assert summary['class_match_rate_denominator'] == 1
    assert summary['score_delta_sample_count'] == 1
    assert summary['class_match_rate'] == 0.0


def test_missing_hypotheses_remain_geometric_but_are_not_score_or_class_samples():
    reference = frame(0, 1, [make_detection_without_results(10, 10, 4, 4)])
    candidate = frame(0, 1, [make_detection_without_results(10, 10, 4, 4)])

    comparison = compare.compare_frame(reference, candidate, class_aware_matching=True)

    assert comparison.matched_count == 1
    assert comparison.mean_iou == 1.0
    assert comparison.mean_score_delta is None
    assert comparison.score_delta_values == ()
    assert comparison.class_match_rate is None
    assert comparison.class_match_count == 0
    assert comparison.class_match_sample_count == 0

    summary = compare.summarize([comparison], 1, 1, 0, 0)
    assert summary['matched_detections'] == 1
    assert summary['aggregate_distributions']['iou']['count'] == 1
    assert summary['aggregate_distributions']['score_delta']['count'] == 0
    assert summary['class_match_rate_denominator'] == 0
    assert summary['class_match_rate'] is None


def test_class_aware_matching_is_an_explicit_diagnostic_option():
    reference = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='cup')])
    candidate = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='box')])

    unconstrained = compare.compare_frame(reference, candidate)
    class_aware = compare.compare_frame(reference, candidate, class_aware_matching=True)

    assert unconstrained.matched_count == 1
    assert unconstrained.class_match_rate == 0.0
    assert class_aware.matched_count == 0
    assert class_aware.unmatched_reference_count == 1
    assert class_aware.unmatched_candidate_count == 1
    assert class_aware.mean_iou is None


def test_class_aware_matching_does_not_treat_missing_class_as_a_wildcard():
    reference = frame(0, 1, [make_detection_without_results(10, 10, 4, 4)])
    candidate = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='cup')])

    unconstrained = compare.compare_frame(reference, candidate)
    class_aware = compare.compare_frame(reference, candidate, class_aware_matching=True)

    assert unconstrained.matched_count == 1
    assert unconstrained.class_match_rate is None
    assert class_aware.matched_count == 0
    assert class_aware.unmatched_count == 2


def test_pair_frames_by_index():
    reference = [frame(0, 10, []), frame(1, 20, [])]
    candidate = [frame(0, 99, [])]
    pairs, unpaired_reference, unpaired_candidate = compare.pair_frames(
        reference, candidate, 'index'
    )
    assert len(pairs) == 1
    assert unpaired_reference == 1
    assert unpaired_candidate == 0


def test_pair_frames_by_stamp_and_preserves_duplicate_fifo():
    reference = [frame(0, 10, []), frame(1, 20, []), frame(2, 10, [])]
    candidate = [frame(0, 10, []), frame(1, 10, []), frame(2, 30, [])]
    pairs, unpaired_reference, unpaired_candidate = compare.pair_frames(
        reference, candidate, 'stamp'
    )
    assert [(ref.index, cand.index) for ref, cand in pairs] == [(0, 0), (2, 1)]
    assert unpaired_reference == 1
    assert unpaired_candidate == 1


def test_zero_paired_frames_have_null_metrics_and_complete_coverage():
    summary = compare.summarize([], 2, 1, 2, 1, 3, 4)

    assert summary['paired_frames'] == 0
    assert summary['unpaired_reference_frames'] == 2
    assert summary['unpaired_candidate_frames'] == 1
    assert summary['unpaired_reference_detections'] == 3
    assert summary['unpaired_candidate_detections'] == 4
    assert summary['mean_iou'] is None
    assert summary['mean_score_delta'] is None
    assert summary['class_match_rate'] is None
    assert summary['frame_coverage']['reference_paired_rate'] == 0.0
    assert summary['frame_coverage']['candidate_paired_rate'] == 0.0
    assert summary['aggregate_distributions']['iou']['count'] == 0
    assert summary['aggregate_distributions']['score_delta']['count'] == 0


def test_class_match_rate_uses_matched_detection_denominator():
    reference_many = frame(
        0,
        1,
        [
            make_detection(10, 10, 4, 4, class_id='cup'),
            make_detection(20, 20, 4, 4, class_id='cup'),
            make_detection(30, 30, 4, 4, class_id='box'),
            make_detection(40, 40, 4, 4, class_id='can'),
        ],
    )
    candidate_many = frame(
        0,
        1,
        [
            make_detection(10, 10, 4, 4, class_id='cup'),
            make_detection(20, 20, 4, 4, class_id='box'),
            make_detection(30, 30, 4, 4, class_id='box'),
            make_detection(40, 40, 4, 4, class_id='other'),
        ],
    )
    reference_one = frame(1, 2, [make_detection(10, 10, 4, 4, class_id='cup')])
    candidate_one = frame(1, 2, [make_detection(10, 10, 4, 4, class_id='cup')])
    comparisons = [
        compare.compare_frame(reference_many, candidate_many),
        compare.compare_frame(reference_one, candidate_one),
    ]

    summary = compare.summarize(comparisons, 2, 2, 0, 0)

    assert summary['class_match_count'] == 3
    assert summary['class_match_rate_denominator'] == 5
    assert summary['class_match_rate'] == pytest.approx(0.6)
    assert summary['aggregate_distributions']['class_match_rate_per_frame']['count'] == 2


def test_worst_frame_details_prioritizes_larger_raw_differences():
    same = frame(0, 1, [make_detection(10, 10, 4, 4, class_id='cup')])
    matching = compare.compare_frame(same, same)
    class_mismatch = compare.compare_frame(
        same,
        frame(1, 1, [make_detection(10, 10, 4, 4, class_id='box')]),
    )

    details = compare.worst_frame_details([matching, class_mismatch], limit=1)

    assert details[0]['reference_index'] == 0
    assert details[0]['class_mismatch_count'] == 1


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


def run_cli(reference_bag, candidate_bag, output_json, *arguments):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--reference-bag',
            str(reference_bag),
            '--candidate-bag',
            str(candidate_bag),
            '--output-json',
            str(output_json),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def load_strict_json(path):
    def reject_constant(value):
        raise AssertionError(f'Non-standard JSON constant: {value}')

    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def test_cli_reports_numeric_differences_and_complete_stamp_coverage(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'

    write_detection_bag(
        reference_bag,
        topic,
        [
            make_msg(10, [make_detection(10, 10, 4, 4, score=0.90, class_id='cup')]),
            make_msg(10, [make_detection(20, 20, 4, 4, score=0.80, class_id='box')]),
            make_msg(20, []),
        ],
    )
    write_detection_bag(
        candidate_bag,
        topic,
        [
            make_msg(10, [make_detection(10, 10, 4, 4, score=0.10, class_id='dog')]),
            make_msg(10, []),
            make_msg(30, [make_detection(30, 30, 4, 4, class_id='can')]),
        ],
    )

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = load_strict_json(output_json)
    assert report['status'] == 'REPORT_ONLY'
    assert report['comparison_schema'] == 'phase2b_detection_report_only_v2'
    assert report['comparison_method']['frame_pairing']['policy'] == 'stamp'
    assert report['comparison_method']['confidence_filter'] == 'none'
    assert report['comparison_method']['top_k_truncation'] == 'none'
    assert report['comparison_method']['detection_pairing']['class_aware'] is False
    assert report['inputs']['reference_bag'] == str(reference_bag)
    assert report['inputs']['candidate_bag'] == str(candidate_bag)
    assert report['inputs']['reference_storage_id'] == 'sqlite3'
    assert report['inputs']['resolved_reference_topic'] == topic
    assert report['inputs']['resolved_candidate_topic'] == topic
    assert report['summary']['paired_frames'] == 2
    assert report['summary']['unpaired_reference_frames'] == 1
    assert report['summary']['unpaired_candidate_frames'] == 1
    assert report['summary']['frame_coverage']['reference_paired_rate'] == pytest.approx(2 / 3)
    assert report['summary']['frame_coverage']['candidate_paired_rate'] == pytest.approx(2 / 3)
    assert report['per_frame'][0]['candidate_index'] == 0
    assert report['per_frame'][0]['class_match_rate'] == 0.0
    assert report['per_frame'][1]['candidate_index'] == 1
    assert report['per_frame'][1]['mean_iou'] is None
    assert report['per_frame'][1]['mean_score_delta'] is None
    assert report['unpaired_frames']['reference'][0]['stamp_ns'] == 20
    assert report['unpaired_frames']['candidate'][0]['stamp_ns'] == 30
    assert report['summary']['class_match_count'] == 0
    assert report['summary']['class_match_rate_denominator'] == 1
    assert report['summary']['class_match_rate'] == 0.0
    assert report['summary']['score_delta_sample_count'] == 1
    assert report['summary']['aggregate_distributions']['iou']['count'] == 1
    assert report['summary']['aggregate_distributions']['score_delta']['count'] == 1
    assert report['worst_frames'][0]['reference_index'] == 1
    assert len(report['comparator_source_sha256']) == 64


def test_cli_empty_bags_report_null_metrics(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    write_detection_bag(reference_bag, topic, [])
    write_detection_bag(candidate_bag, topic, [])

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = load_strict_json(output_json)
    assert report['status'] == 'REPORT_ONLY'
    assert report['summary']['paired_frames'] == 0
    assert report['summary']['mean_iou'] is None
    assert report['summary']['mean_score_delta'] is None
    assert report['summary']['class_match_rate'] is None
    assert report['summary']['aggregate_distributions']['iou']['count'] == 0
    assert report['unpaired_frames'] == {'reference': [], 'candidate': []}


def test_cli_zero_stamp_pairs_preserve_each_unpaired_frame(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    write_detection_bag(reference_bag, topic, [make_msg(10, [])])
    write_detection_bag(candidate_bag, topic, [make_msg(20, [])])

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = load_strict_json(output_json)
    assert report['summary']['paired_frames'] == 0
    assert report['summary']['unpaired_reference_frames'] == 1
    assert report['summary']['unpaired_candidate_frames'] == 1
    assert report['summary']['mean_iou'] is None
    assert report['summary']['class_match_rate'] is None
    assert report['unpaired_frames']['reference'][0]['stamp_ns'] == 10
    assert report['unpaired_frames']['candidate'][0]['stamp_ns'] == 20


def test_cli_records_explicit_diagnostic_options(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    msg = make_msg(
        10,
        [
            make_detection(10, 10, 4, 4, score=0.9, class_id='cup'),
            make_detection(10, 10, 4, 4, score=0.1, class_id='box'),
        ],
    )
    write_detection_bag(reference_bag, topic, [msg])
    write_detection_bag(
        candidate_bag,
        topic,
        [make_msg(99, [make_detection(10, 10, 4, 4, score=0.9, class_id='box')])],
    )

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--match-policy',
        'index',
        '--storage-id',
        'sqlite3',
        '--min-score',
        '0.5',
        '--max-detections-per-frame',
        '1',
        '--class-aware-matching',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = load_strict_json(output_json)
    assert report['comparison_method']['frame_pairing']['policy'] == 'index'
    assert 'bag-order index pairing' in report['comparison_method']['frame_pairing']['description']
    assert report['comparison_method']['confidence_filter'] == 'score >= 0.5'
    assert report['comparison_method']['top_k_truncation'] == 'top 1 detections by score'
    assert report['comparison_method']['detection_pairing']['class_aware'] is True
    assert report['filters'] == {'min_score': 0.5, 'max_detections_per_frame': 1}
    assert report['per_frame'][0]['matched_count'] == 0


@pytest.mark.parametrize(
    'obsolete_option',
    ['--report-only', '--ignore-unpaired-frames', '--min-mean-iou'],
)
def test_cli_rejects_removed_numeric_gate_options(tmp_path, obsolete_option):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--reference-bag',
            str(tmp_path / 'reference'),
            '--candidate-bag',
            str(tmp_path / 'candidate'),
            '--output-json',
            str(tmp_path / 'report.json'),
            obsolete_option,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert 'unrecognized arguments' in result.stderr


def test_cli_nonfinite_score_is_error_even_when_filter_would_drop_detection(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    write_detection_bag(
        reference_bag,
        topic,
        [make_msg(10, [make_detection(10, 10, 4, 4, score=float('nan'))])],
    )
    write_detection_bag(candidate_bag, topic, [])

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
        '--min-score',
        '0.5',
    )

    assert result.returncode == 1
    report = load_strict_json(output_json)
    assert report['status'] == 'ERROR'
    assert 'score must be finite' in report['error']


def test_cli_nonfinite_bbox_is_error_even_when_filter_would_drop_detection(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    invalid = make_detection(10, 10, 4, 4, score=0.1)
    invalid.bbox.size_x = float('nan')
    write_detection_bag(reference_bag, topic, [make_msg(10, [invalid])])
    write_detection_bag(candidate_bag, topic, [])

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
        '--min-score',
        '0.5',
    )

    assert result.returncode == 1
    report = load_strict_json(output_json)
    assert report['status'] == 'ERROR'
    assert 'bounding box values must be finite' in report['error']


def test_cli_bag_and_topic_errors_remain_nonzero(tmp_path):
    topic = '/detections_output'
    valid_bag = tmp_path / 'valid'
    missing_bag = tmp_path / 'missing'
    write_detection_bag(valid_bag, topic, [make_msg(10, [])])

    missing_result = run_cli(
        missing_bag,
        valid_bag,
        tmp_path / 'missing-report.json',
        '--storage-id',
        'sqlite3',
    )
    assert missing_result.returncode == 1
    assert load_strict_json(tmp_path / 'missing-report.json')['status'] == 'ERROR'

    wrong_topic_result = run_cli(
        valid_bag,
        valid_bag,
        tmp_path / 'topic-report.json',
        '--storage-id',
        'sqlite3',
        '--reference-topic',
        '/not_a_detection_topic',
    )
    assert wrong_topic_result.returncode == 1
    topic_report = load_strict_json(tmp_path / 'topic-report.json')
    assert topic_report['status'] == 'ERROR'
    assert 'not found in bag' in topic_report['error']


def test_cli_missing_hypotheses_have_geometry_only_metrics(tmp_path):
    topic = '/detections_output'
    reference_bag = tmp_path / 'reference'
    candidate_bag = tmp_path / 'candidate'
    output_json = tmp_path / 'report.json'
    write_detection_bag(
        reference_bag,
        topic,
        [make_msg(10, [make_detection_without_results(10, 10, 4, 4)])],
    )
    write_detection_bag(
        candidate_bag,
        topic,
        [make_msg(10, [make_detection_without_results(10, 10, 4, 4)])],
    )

    result = run_cli(
        reference_bag,
        candidate_bag,
        output_json,
        '--storage-id',
        'sqlite3',
        '--class-aware-matching',
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = load_strict_json(output_json)
    comparison = report['per_frame'][0]
    assert comparison['matched_count'] == 1
    assert comparison['mean_iou'] == 1.0
    assert comparison['mean_score_delta'] is None
    assert comparison['score_delta_sample_count'] == 0
    assert comparison['class_match_rate'] is None
    assert comparison['class_match_sample_count'] == 0
    assert report['summary']['aggregate_distributions']['score_delta']['count'] == 0
    assert report['summary']['class_match_rate_denominator'] == 0
