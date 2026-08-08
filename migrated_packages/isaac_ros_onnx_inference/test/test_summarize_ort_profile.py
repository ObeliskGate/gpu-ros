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

"""Tests for the ONNX Runtime profile provider summary."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'summarize_ort_profile.py'
SPEC = importlib.util.spec_from_file_location('summarize_ort_profile', SCRIPT_PATH)
PROFILE_SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE_SUMMARY)


def write_profile(path, events):
    """Write a minimal ORT profile fixture."""
    path.write_text(json.dumps(events), encoding='utf-8')


def test_reports_accelerator_without_cpu_fallback(tmp_path):
    """A profile with only MIGraphX kernels has no CPU provider entry."""
    profile_path = tmp_path / 'migraphx.json'
    write_profile(profile_path, [{
        'cat': 'Node',
        'name': 'fused_kernel_time',
        'dur': 12.5,
        'args': {
            'provider': 'MIGraphXExecutionProvider',
            'op_name': 'MIGraphX',
        },
    }])

    report = PROFILE_SUMMARY.summarize_profile(profile_path)

    assert report['provider_kernel_events_found'] is True
    assert report['cpu_fallback_observed'] is False
    assert 'MIGraphXExecutionProvider' in report['providers']
    assert PROFILE_SUMMARY.CPU_PROVIDER not in report['providers']


def test_reports_cpu_node_names(tmp_path):
    """CPU provider kernel events retain node and operator names."""
    profile_path = tmp_path / 'mixed.json'
    write_profile(profile_path, [{
        'cat': 'Node',
        'name': 'shape_kernel_time',
        'dur': 2.0,
        'args': {
            'provider': 'CPUExecutionProvider',
            'op_name': 'Shape',
        },
    }])

    report = PROFILE_SUMMARY.summarize_profile(profile_path)
    cpu_data = report['providers'][PROFILE_SUMMARY.CPU_PROVIDER]

    assert report['cpu_fallback_observed'] is True
    assert cpu_data['kernel_events'] == 1
    assert cpu_data['duration_us'] == 2.0
    assert cpu_data['unique_nodes'] == ['shape (Shape)']


def test_ignores_non_provider_events(tmp_path):
    """Session-level timing events are not mistaken for graph nodes."""
    profile_path = tmp_path / 'session-only.json'
    write_profile(profile_path, [{
        'cat': 'Session',
        'name': 'session_initialization',
        'dur': 100.0,
        'args': {},
    }])

    report = PROFILE_SUMMARY.summarize_profile(profile_path)

    assert report['provider_kernel_events_found'] is False
    assert report['providers'] == {}


def test_provider_layout_comparison_ignores_event_counts(tmp_path):
    """Equivalent provider placement matches even when sample counts differ."""
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    cuda_event = {
        'cat': 'Node',
        'name': 'conv_kernel_time',
        'dur': 3.0,
        'args': {
            'provider': 'CUDAExecutionProvider',
            'op_name': 'Conv',
        },
    }
    write_profile(first, [cuda_event])
    write_profile(second, [cuda_event, cuda_event])

    comparison = PROFILE_SUMMARY.compare_provider_layouts([
        PROFILE_SUMMARY.summarize_profile(first),
        PROFILE_SUMMARY.summarize_profile(second),
    ])

    assert comparison['match'] is True
    assert comparison['differences'] == []


def test_provider_layout_comparison_reports_extra_cpu_node(tmp_path):
    """A candidate-only CPU fallback node makes the layouts differ."""
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    cuda_event = {
        'cat': 'Node',
        'name': 'conv_kernel_time',
        'dur': 3.0,
        'args': {
            'provider': 'CUDAExecutionProvider',
            'op_name': 'Conv',
        },
    }
    cpu_event = {
        'cat': 'Node',
        'name': 'shape_kernel_time',
        'dur': 1.0,
        'args': {
            'provider': 'CPUExecutionProvider',
            'op_name': 'Shape',
        },
    }
    write_profile(first, [cuda_event])
    write_profile(second, [cuda_event, cpu_event])

    comparison = PROFILE_SUMMARY.compare_provider_layouts([
        PROFILE_SUMMARY.summarize_profile(first),
        PROFILE_SUMMARY.summarize_profile(second),
    ])

    assert comparison['match'] is False
    assert comparison['differences'][0]['extra_in_candidate'] == {
        'CPUExecutionProvider': ['shape (Shape)'],
    }


def test_cpu_layout_difference_can_be_ignored_for_closure(tmp_path):
    """Known CPU fallback remains visible but does not fail accelerator closure."""
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    cuda_event = {
        'cat': 'Node',
        'name': 'conv_kernel_time',
        'dur': 3.0,
        'args': {'provider': 'CUDAExecutionProvider', 'op_name': 'Conv'},
    }
    cpu_event = {
        'cat': 'Node',
        'name': 'shape_kernel_time',
        'dur': 1.0,
        'args': {'provider': 'CPUExecutionProvider', 'op_name': 'Shape'},
    }
    write_profile(first, [cuda_event])
    write_profile(second, [cuda_event, cpu_event])

    comparison = PROFILE_SUMMARY.compare_provider_layouts(
        [PROFILE_SUMMARY.summarize_profile(first), PROFILE_SUMMARY.summarize_profile(second)],
        ignored_providers={PROFILE_SUMMARY.CPU_PROVIDER},
    )

    assert comparison['match'] is True
    assert PROFILE_SUMMARY.CPU_PROVIDER in (
        PROFILE_SUMMARY.summarize_profile(second)['providers'])
