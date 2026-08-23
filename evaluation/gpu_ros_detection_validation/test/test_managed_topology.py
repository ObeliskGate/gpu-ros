# Copyright 2026 Boshen Chen

import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
ONNX_LAUNCH = ROOT / 'migrated_packages' / 'gpu_ros_onnx_inference' / 'launch'
BENCHMARKS = ROOT / 'migrated_packages' / 'benchmarks'


def test_production_managed_launches_have_no_staging_plugins():
    for name in ('yolov8_ort_managed_amd.launch.py', 'rtdetr_ort_managed_amd.launch.py'):
        text = (ONNX_LAUNCH / name).read_text()
        assert 'hip_managed_strict' in text
        assert 'StdToManagedHipTensorBundleNode' not in text
        assert 'ManagedHipToStdTensorBundleNode' not in text


def test_production_managed_launch_plugin_order_is_direct():
    yolo = (ONNX_LAUNCH / 'yolov8_ort_managed_amd.launch.py').read_text()
    assert yolo.count('YoloV8ManagedHipImageEncoderNode') == 1
    assert yolo.count('YoloV8ManagedHipDecoderNode') == 1
    assert yolo.count('OnnxInferenceNode') == 1
    assert yolo.index('YoloV8ManagedHipImageEncoderNode') < yolo.index(
        'OnnxInferenceNode') < yolo.index('YoloV8ManagedHipDecoderNode')
    rtdetr = (ONNX_LAUNCH / 'rtdetr_ort_managed_amd.launch.py').read_text()
    assert rtdetr.count('RtDetrManagedHipImageEncoderNode') == 1
    assert rtdetr.count('RtDetrManagedHipPreprocessorNode') == 1
    assert rtdetr.count('RtDetrManagedHipDecoderNode') == 1
    assert rtdetr.count('OnnxInferenceNode') == 1
    assert rtdetr.index('RtDetrManagedHipImageEncoderNode') < rtdetr.index(
        'RtDetrManagedHipPreprocessorNode') < rtdetr.index('OnnxInferenceNode')
    assert rtdetr.index('OnnxInferenceNode') < rtdetr.index(
        'RtDetrManagedHipDecoderNode')
    assert "('managed_tensor_output', 'managed_tensor_image')" in rtdetr
    assert "('managed_tensor_input', 'managed_tensor_image')" in rtdetr
    assert "('tensor_output', 'managed_tensor_output_ort')" in rtdetr


def test_managed_benchmarks_are_direct_and_strict():
    expected = {
        'gpu_ros_yolov8_phase2b_amd_managed_graph.py': (
            'YoloV8ManagedHipImageEncoderNode',
            'YoloV8ManagedHipDecoderNode',
        ),
        'gpu_ros_rtdetr_phase2b_amd_managed_graph.py': (
            'RtDetrManagedHipImageEncoderNode',
            'RtDetrManagedHipPreprocessorNode',
            'RtDetrManagedHipDecoderNode',
        ),
    }
    for name, plugins in expected.items():
        text = (BENCHMARKS / name).read_text()
        assert 'hip_managed_strict' in text
        assert 'StdToManagedHipTensorBundleNode' not in text
        assert 'ManagedHipToStdTensorBundleNode' not in text
        assert 'additional_fixed_publisher_rate_tests=[10.0, 30.0, 60.0]' in text
        for plugin in plugins:
            assert text.count(plugin) == 1


def test_staged_control_benchmarks_contain_only_the_explicit_adapter_lane():
    for name in (
        'gpu_ros_yolov8_phase2b_amd_staged_control_graph.py',
        'gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py',
    ):
        text = (BENCHMARKS / name).read_text()
        assert 'hip_managed_strict' in text
        assert text.count('ManagedHipToStdTensorBundleNode') >= 2
        assert text.count('StdToManagedHipTensorBundleNode') >= 1
        assert 'staged control' in text.lower()
        assert 'additional_fixed_publisher_rate_tests=[10.0, 30.0, 60.0]' in text


def test_phase2a_reference_benchmarks_use_the_same_fixed_rate_trials():
    for name in (
        'gpu_ros_yolov8_phase2a_amd_graph.py',
        'gpu_ros_rtdetr_phase2a_amd_graph.py',
    ):
        text = (BENCHMARKS / name).read_text()
        assert 'additional_fixed_publisher_rate_tests=[10.0, 30.0, 60.0]' in text


def test_phase2b_benchmark_modules_expose_one_test_class_each():
    """Prevent imported benchmark base classes from running a second sweep."""
    expected = {
        'gpu_ros_yolov8_phase2b_amd_managed_graph.py':
            'TestGpuRosYoloV8Phase2bAmdManaged',
        'gpu_ros_yolov8_phase2b_amd_staged_control_graph.py':
            'TestGpuRosYoloV8Phase2bAmdStagedControl',
        'gpu_ros_rtdetr_phase2b_amd_managed_graph.py':
            'TestGpuRosRtDetrPhase2bAmdManaged',
        'gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py':
            'TestGpuRosRtDetrPhase2bAmdStagedControl',
    }
    for name, expected_class in expected.items():
        tree = ast.parse((BENCHMARKS / name).read_text(), filename=name)
        test_classes = [
            node.name for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test')
        ]
        assert test_classes == [expected_class]
