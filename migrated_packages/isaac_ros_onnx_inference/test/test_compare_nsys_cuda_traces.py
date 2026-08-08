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

"""Tests for the Nsight Systems CUDA trace comparison."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'compare_nsys_cuda_traces.py'
SPEC = importlib.util.spec_from_file_location('compare_nsys_cuda_traces', SCRIPT_PATH)
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


def test_nsys_memory_units_are_converted_to_exact_bytes():
    """Nsight's unit-qualified byte columns are recognized."""
    assert TRACE_COMPARE.byte_count({'Bytes (B)': 4915200}) == 4915200
    assert TRACE_COMPARE.byte_count({'Bytes (MB)': 4.9152}) == 4915200
    assert TRACE_COMPARE.byte_count({'Bytes (MiB)': 1.5}) == 1572864


def test_equal_kernel_and_memcpy_traces_pass():
    """Identical GPU work satisfies the transport audit."""
    events = [
        kernel('conv_kernel'),
        memcpy('[CUDA memcpy DtoH]', 2822400, 'Device', 'Pageable'),
    ]

    result = TRACE_COMPARE.compare(
        events, events, {2822400, 4915200}, 100, 100)

    assert result['pass'] is True
    assert result['bridge_zero_copy_pass'] is True
    assert result['criteria']['kernel_name_sets_match'] is True
    assert result['memcpy']['managed_extra_signatures'] == []


def test_managed_only_payload_copy_fails():
    """An M-only tensor-sized memcpy is reported and fails."""
    config_c = [kernel('conv_kernel')]
    managed = [
        kernel('conv_kernel'),
        memcpy('[CUDA memcpy DtoD]', 4915200, 'Device', 'Device'),
    ]

    result = TRACE_COMPARE.compare(
        config_c, managed, {2822400, 4915200}, 100, 100)

    assert result['pass'] is False
    assert result['bridge_zero_copy_pass'] is False
    assert result['criteria']['managed_has_no_extra_memcpy_signature'] is False
    assert result['payload_copy_counts']['deltas'][0]['delta'] == 1


def test_one_extra_frame_does_not_look_like_a_new_copy_path():
    """Per-frame normalization tolerates one capture-boundary frame."""
    config_c = [
        memcpy('[CUDA memcpy DtoH]', 2822400, 'Device', 'Pageable')
        for _ in range(393)
    ]
    managed = [
        memcpy('[CUDA memcpy DtoH]', 2822400, 'Device', 'Pageable')
        for _ in range(394)
    ]

    result = TRACE_COMPARE.compare(
        config_c, managed, {2822400}, 393, 394)

    assert result['pass'] is True
    assert result['criteria']['managed_has_no_extra_payload_copy_rate'] is True


def test_same_signature_with_one_extra_payload_copy_per_frame_fails():
    """An existing copy signature cannot hide a Managed bridge copy."""
    config_c = [
        memcpy('[CUDA memcpy DtoD]', 4915200, 'Device', 'Device')
        for _ in range(10)
    ]
    managed = [
        memcpy('[CUDA memcpy DtoD]', 4915200, 'Device', 'Device')
        for _ in range(20)
    ]

    result = TRACE_COMPARE.compare(
        config_c, managed, {4915200}, 10, 10)

    assert result['pass'] is False
    assert result['criteria']['managed_has_no_extra_memcpy_signature'] is True
    assert result['criteria']['managed_has_no_extra_payload_copy_rate'] is False


def test_managed_only_ordinary_kernel_is_diagnostic_only():
    """A provider kernel difference is not itself a transport copy failure."""
    result = TRACE_COMPARE.compare(
        [kernel('conv_kernel')],
        [kernel('conv_kernel'), kernel('unexpected_kernel')],
        set(),
    )

    assert result['pass'] is True
    assert result['status'] == 'PASS'
    assert result['bridge_zero_copy_pass'] is True
    assert result['gpu_execution_control_pass'] is False
    assert result['kernel_names']['extra_in_managed'] == ['unexpected_kernel']
    assert result['kernel_only_differences'][0]['classification'].startswith('diagnostic_')


def test_managed_only_copy_kernel_is_inconclusive():
    """A copy-looking kernel needs payload evidence, but is not an automatic FAIL."""
    result = TRACE_COMPARE.compare(
        [kernel('conv_kernel')],
        [kernel('conv_kernel'), kernel('payload_copy_kernel')],
        set(),
    )

    assert result['pass'] is False
    assert result['status'] == 'INCONCLUSIVE'
    assert result['memory_copy_failures'] == []
    assert result['unresolved_payload_copy_risk'][0]['name'] == 'payload_copy_kernel'


def test_pointer_lifetime_confirmed_boundary_copy_fails():
    """Independent pointer/lifetime evidence can promote a trace to FAIL."""
    result = TRACE_COMPARE.compare(
        [kernel('conv_kernel')],
        [kernel('conv_kernel')],
        set(),
        boundary_evidence={'managed_boundary_payload_copy_confirmed': True},
    )

    assert result['status'] == 'FAIL'
    assert result['memory_copy_failures'][0]['reason'].startswith('pointer/lifetime')
