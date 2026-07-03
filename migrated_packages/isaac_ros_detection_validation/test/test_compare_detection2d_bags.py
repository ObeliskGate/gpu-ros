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


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert compare.iou(box, box) == 1.0


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


def write_detection_bag(path, topic, messages):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

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
            '--reference-topic', topic,
            '--candidate-topic', topic,
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
