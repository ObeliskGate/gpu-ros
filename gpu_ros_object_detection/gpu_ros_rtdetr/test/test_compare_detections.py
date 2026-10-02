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
from pathlib import Path

import pytest
import rclpy
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_detections.py'
SPEC = importlib.util.spec_from_file_location('compare_detections', SCRIPT_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def make_detection(score=0.9, class_id='0'):
    detection = Detection2D()
    detection.bbox.size_x = 10.0
    detection.bbox.size_y = 10.0
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = class_id
    result.hypothesis.score = score
    detection.results.append(result)
    return detection


def make_message(stamp, detections):
    message = Detection2DArray()
    message.header.stamp.sec = stamp
    message.detections = detections
    return message


@pytest.fixture
def comparator():
    rclpy.init(args=[])
    node = compare.DetectionComparator('/test/a', '/test/b')
    try:
        yield node
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_zero_stamp_overlap_reports_pending_coverage_and_unavailable_metrics(comparator):
    comparator._on(make_message(1, [make_detection()]), comparator.buf_a, comparator.buf_b)
    comparator._on(
        make_message(2, [make_detection(), make_detection()]),
        comparator.buf_b,
        comparator.buf_a,
    )

    summary = comparator.summary()

    assert summary['frames'] == {
        'received_a': 1,
        'received_b': 1,
        'paired': 0,
        'unmatched_a': 1,
        'unmatched_b': 1,
        'pending_a': 1,
        'pending_b': 1,
    }
    assert summary['detections']['pending_a'] == 1
    assert summary['detections']['pending_b'] == 2
    assert summary['detections']['unmatched_a'] == 1
    assert summary['detections']['unmatched_b'] == 2
    assert summary['iou'] == {'samples': 0, 'mean': None, 'median': None, 'p95': None}
    assert summary['score_delta'] == {
        'samples': 0,
        'mean': None,
        'median': None,
        'p95': None,
    }
    assert summary['class_observations']['match_rate'] is None


def test_geometry_matches_report_score_and_class_observations(comparator):
    comparator._on(
        make_message(3, [make_detection(score=0.8, class_id='person')]),
        comparator.buf_a,
        comparator.buf_b,
    )
    comparator._on(
        make_message(3, [make_detection(score=0.6, class_id='vehicle')]),
        comparator.buf_b,
        comparator.buf_a,
    )

    summary = comparator.summary()

    assert summary['frames']['paired'] == 1
    assert summary['frames']['unmatched_a'] == 0
    assert summary['frames']['unmatched_b'] == 0
    assert summary['detections']['matched'] == 1
    assert summary['detections']['unmatched_a'] == 0
    assert summary['detections']['unmatched_b'] == 0
    assert summary['iou']['samples'] == 1
    assert summary['iou']['mean'] == pytest.approx(1.0)
    assert summary['score_delta']['samples'] == 1
    assert summary['score_delta']['mean'] == pytest.approx(0.2)
    assert summary['class_observations']['matches'] == 0
    assert summary['class_observations']['mismatches'] == 1
    assert summary['class_observations']['match_rate'] == pytest.approx(0.0)


def test_replaced_duplicate_stamp_remains_in_unpaired_coverage(comparator):
    comparator._on(make_message(1, [make_detection()]), comparator.buf_a, comparator.buf_b)
    comparator._on(
        make_message(1, [make_detection(), make_detection()]),
        comparator.buf_a,
        comparator.buf_b,
    )
    comparator._on(make_message(1, [make_detection()]), comparator.buf_b, comparator.buf_a)

    summary = comparator.summary()

    assert summary['frames']['received_a'] == 2
    assert summary['frames']['paired'] == 1
    assert summary['frames']['unmatched_a'] == 1
    assert summary['frames']['pending_a'] == 0
    assert summary['detections']['received_a'] == 3
    assert summary['detections']['matched'] == 1
    assert summary['detections']['unmatched_paired_a'] == 1
    assert summary['detections']['unpaired_a'] == 1
