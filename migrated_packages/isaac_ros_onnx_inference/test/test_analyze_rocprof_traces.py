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

"""Tests for ROCprofiler JSON normalization and AMD copy closure rules."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'analyze_rocprof_traces.py'
SPEC = importlib.util.spec_from_file_location('analyze_rocprof_traces', SCRIPT_PATH)
ROC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROC)


def memory_copy(operation, size, source, destination, frame=None):
    record = {
        'kind': 'memory_copy',
        'operation': operation,
        'bytes': size,
        'src_agent': source,
        'dst_agent': destination,
    }
    if frame is not None:
        record['frame_id'] = frame
    return record


def kernel(name):
    return {'kind': 'kernel_dispatch', 'kernel_name': name}


def kernel_with_bytes(name, size):
    return {'kind': 'kernel_dispatch', 'kernel_name': name, 'bytes': size}


def test_rocprof_directions_bytes_and_frame_normalization():
    events = [
        memory_copy('hipMemcpyHtoD', 4915200, 'CPU agent', 'GPU agent', frame=1),
        memory_copy('hipMemcpyDtoH', '2.8224 MB', 'GPU agent', 'CPU agent', frame=1),
        memory_copy('hipMemcpyDtoD', 1600, 'GPU0', 'GPU0', frame=2),
    ]

    report = ROC.summarize(events, {4915200, 2822400, 1600}, frames=2)

    assert report['memory_totals']['H2D'] == {'count': 1, 'bytes': 4915200}
    assert report['memory_totals']['D2H'] == {'count': 1, 'bytes': 2822400}
    assert report['memory_totals']['D2D'] == {'count': 1, 'bytes': 1600}
    assert report['normalized_per_frame']['H2D']['count'] == 0.5
    assert report['payload_copy_counts'][(1600, 'D2D')] == 1


def test_rocprof_multi_process_wrapper_is_flattened(tmp_path):
    first = tmp_path / 'pid-1.json'
    second = tmp_path / 'pid-2.json'
    first.write_text(json.dumps({'pid': 11, 'records': [kernel('a')]}), encoding='utf-8')
    second.write_text(json.dumps({'pid': 12, 'records': [
        memory_copy('hipMemcpyHtoD', 16, 'host', 'device'),
    ]}), encoding='utf-8')

    events = ROC.load_events([first, second])
    assert len(events) == 2
    report = ROC.summarize(events, {16})
    assert report['kernel_event_count'] == 1
    assert report['memory_totals']['H2D']['bytes'] == 16


def test_managed_only_tensor_sized_d2d_copy_fails():
    result = ROC.compare(
        [kernel('provider_kernel')],
        [kernel('provider_kernel'), memory_copy('hipMemcpyDtoD', 4915200, 'GPU', 'GPU')],
        {4915200},
        std_frames=10,
        managed_frames=10,
    )

    assert result['status'] == 'FAIL'
    assert result['memory_copy_failures'][0]['reason'].startswith('managed-only tensor-sized')


def test_managed_adapter_h2d_d2h_is_recorded_but_not_boundary_failure():
    result = ROC.compare(
        [kernel('std_provider_kernel')],
        [
            memory_copy('hipMemcpyHtoD', 4915200, 'CPU', 'GPU'),
            memory_copy('hipMemcpyDtoH', 2822400, 'GPU', 'CPU'),
        ],
        {4915200, 2822400},
        std_frames=10,
        managed_frames=10,
    )

    assert result['status'] == 'PASS'
    assert {item['direction'] for item in result['adapter_copy_evidence']} == {'H2D', 'D2H'}
    assert result['memory_copy_failures'] == []


def test_copy_named_kernel_is_inconclusive_but_ordinary_kernel_is_not_fail():
    ordinary = ROC.compare([kernel('a')], [kernel('a'), kernel('provider_fused')], set())
    uncertain = ROC.compare([kernel('a')], [kernel('a'), kernel('blit_candidate')], set())

    assert ordinary['status'] == 'PASS'
    assert uncertain['status'] == 'INCONCLUSIVE'
    assert uncertain['memory_copy_failures'] == []


def test_copy_named_kernel_with_byte_column_is_not_reclassified_as_memory_copy():
    result = ROC.compare(
        [kernel('a')],
        [kernel('a'), kernel_with_bytes('payload_copy_kernel', 4915200)],
        {4915200},
    )

    assert result['status'] == 'INCONCLUSIVE'
    assert result['memory_copy_failures'] == []
    assert result['managed']['memory_copy'] == []


def test_uninterpretable_profiler_records_are_inconclusive():
    result = ROC.compare(
        [{'kind': 'runtime_api', 'bytes': 7}],
        [kernel('provider_kernel')],
        set(),
    )

    assert result['status'] == 'INCONCLUSIVE'
    assert result['criteria']['profiler_data_complete'] is False


def test_pointer_lifetime_boundary_evidence_fails():
    binding = {
        'first_frame': True,
        'inputs': [{
            'name': 'input', 'bytes': 16, 'storage': 'hip_device',
            'ort_pointer': '0x1', 'pointer_identity': True,
            'lifetime_path': 'lease',
        }],
        'outputs': [{
            'name': 'output', 'bytes': 16, 'storage': 'hip_device',
            'ort_pointer': '0x2', 'pointer_identity': True,
            'lifetime_path': 'adoption',
        }],
    }
    result = ROC.compare(
        [], [], set(), boundary_evidence={
            'managed_boundary_payload_copy_confirmed': True,
        }, binding_reports={'reference': binding, 'managed': binding})

    assert result['status'] == 'FAIL'
    assert result['pointer_lifetime_evidence']['complete'] is True
