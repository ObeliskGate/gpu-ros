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

"""Tests for the Config C versus Config D CUDA copy delta report."""

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'compare_nvidia_copy_traces.py'
SPEC = importlib.util.spec_from_file_location('compare_nvidia_copy_traces', SCRIPT_PATH)
TRACE_COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE_COMPARE)


def kernel(name):
    """Create one minimal nsys kernel row."""
    return {'Name': name, 'Bytes': ''}


def memcpy(name, size, source, destination):
    """Create one minimal nsys memory-operation row."""
    return {
        'Name': name,
        'Bytes': size,
        'SrcMemKd': source,
        'DstMemKd': destination,
    }


def binding_report():
    """Create the minimum valid first-frame binding evidence."""
    return {
        'first_frame': True,
        'inputs': [{
            'name': 'images',
            'bytes': 4915200,
            'storage': 'cuda_device',
            'lifetime_path': 'lease through ORT Run',
        }],
        'outputs': [{
            'name': 'scores',
            'bytes': 1200,
            'storage': 'cuda_device',
            'lifetime_path': 'ORT-owned output adoption',
        }],
    }


def test_binding_sizes_are_used_as_exact_payload_sizes():
    """Binding bytes, including small control tensors, are not a threshold."""
    report = binding_report()
    assert TRACE_COMPARE._binding_payload_sizes(report) == {4915200, 1200}


def test_positive_h2d_and_d2h_delta_is_reported_per_frame():
    """Explicit host/device records prove an additional D transfer path."""
    reference = [kernel('inference_kernel')]
    candidate = [
        kernel('inference_kernel'),
        memcpy('[CUDA memcpy HtoD]', 4915200, 'Pageable', 'Device'),
        memcpy('[CUDA memcpy DtoH]', 1200, 'Device', 'Pageable'),
    ]
    result = TRACE_COMPARE.compare(
        reference,
        candidate,
        {4915200, 1200},
        10,
        10,
        binding_reports={'reference': binding_report(), 'candidate': binding_report()},
    )

    deltas = {row['direction']: row for row in result['memory_total_deltas']}
    assert deltas['H2D']['byte_delta'] == 4915200
    assert deltas['D2H']['byte_delta'] == 1200
    assert deltas['H2D']['byte_delta_per_frame'] == 491520.0
    assert result['host_device_copy_evidence']['candidate_more_h2d_or_d2h'] is True
    assert result['status'] == 'PASS'


def test_copy_named_kernel_is_not_a_memory_copy_record():
    """A D-only provider kernel is diagnostic until explicit copy evidence exists."""
    result = TRACE_COMPARE.compare(
        [kernel('inference_kernel')],
        [kernel('inference_kernel'), kernel('copy_like_provider_kernel')],
        set(),
        10,
        10,
        binding_reports={'reference': binding_report(), 'candidate': binding_report()},
    )

    assert result['memory_totals']['config_d'] == {}
    assert result['host_device_copy_evidence']['candidate_more_h2d_or_d2h'] is False
    assert result['kernel_only_differences'][0]['name'] == 'copy_like_provider_kernel'
    assert result['unresolved_payload_copy_risk'][0]['name'] == 'copy_like_provider_kernel'
