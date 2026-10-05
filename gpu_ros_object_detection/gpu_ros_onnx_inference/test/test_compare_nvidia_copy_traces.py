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


def binding_report(transport='tensor_list', direct_output=True):
    """Create a fixed RT-DETR first-frame binding report."""
    return {
        'first_frame': True,
        'transport': transport,
        'inputs': [
            {
                'name': 'images',
                'bytes': 4915200,
                'storage': 'cuda_device',
                'input_pointer': '0x1000',
                'ort_pointer': '0x1000',
                'pointer_identity': True,
                'lifetime_path': 'native TensorList input lease through ORT Run',
            },
            {
                'name': 'orig_target_sizes',
                'bytes': 16,
                'storage': 'cpu_buffer',
                'input_pointer': '0x2000',
                'ort_pointer': '0x3000',
                'pointer_identity': False,
                'lifetime_path': 'CPU input promoted into CUDA Buffer',
            },
        ],
        'outputs': [
            {
                'name': name,
                'bytes': size,
                'storage': 'cuda_device' if direct_output else 'cpu_buffer',
                'output_pointer': '0x4000' if direct_output else '0x5000',
                'ort_pointer': '0x4000' if direct_output else '0x4001',
                'pointer_identity': direct_output,
                'lifetime_path': (
                    'native CUDA Buffer writer finalized after IoBinding::SynchronizeOutputs'
                    if direct_output
                    else 'standard TensorBundle output materialization'
                ),
            }
            for name, size in (('labels', 800), ('boxes', 1600), ('scores', 400))
        ],
    }


def test_binding_sizes_include_small_control_and_output_tensors():
    """Binding bytes include orig_target_sizes and all formal RT-DETR outputs."""
    report = binding_report()
    assert TRACE_COMPARE._binding_payload_sizes(report) == {
        16,
        400,
        800,
        1600,
        4915200,
    }


def test_pointer_address_accepts_only_positive_numeric_addresses():
    """Integer, decimal, and hexadecimal address forms normalize exactly."""
    assert TRACE_COMPARE.pointer_address(4096) == 4096
    assert TRACE_COMPARE.pointer_address('0x1000') == 4096
    assert TRACE_COMPARE.pointer_address('4096') == 4096
    assert TRACE_COMPARE.pointer_address('0X1000') == 4096

    for invalid in (
        True,
        None,
        0,
        '0',
        -1,
        '-0x1',
        'unknown',
        1.5,
        {'address': '0x1000'},
    ):
        assert TRACE_COMPARE.pointer_address(invalid) is None


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
        binding_reports={
            'reference': binding_report(),
            'candidate': binding_report('std', direct_output=False),
        },
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
        binding_reports={
            'reference': binding_report(),
            'candidate': binding_report('std', direct_output=False),
        },
    )

    assert result['memory_totals']['config_d'] == {}
    assert result['host_device_copy_evidence']['candidate_more_h2d_or_d2h'] is False
    assert result['kernel_only_differences'][0]['name'] == 'copy_like_provider_kernel'
    assert result['unresolved_payload_copy_risk'][0]['name'] == 'copy_like_provider_kernel'


def test_nonidentical_native_output_pointer_is_inconclusive():
    """A native C output without direct ORT identity cannot close the audit."""
    result = TRACE_COMPARE.compare(
        [kernel('inference_kernel')],
        [kernel('inference_kernel')],
        {4915200, 16, 800, 1600, 400},
        10,
        10,
        binding_reports={
            'reference': binding_report('tensor_list', direct_output=False),
            'candidate': binding_report('std', direct_output=False),
        },
    )

    assert result['schema_version'] == 1
    assert result['status'] == 'INCONCLUSIVE'
    assert result['binding_report_complete'] is False


def test_native_output_integer_and_hex_pointers_remain_equivalent_in_c_vs_d():
    """A positive integer pointer matches the same hexadecimal address."""
    reference = binding_report()
    reference['outputs'][0]['output_pointer'] = 16384
    reference['outputs'][0]['ort_pointer'] = '0x4000'
    result = TRACE_COMPARE.compare(
        [kernel('inference_kernel')],
        [kernel('inference_kernel')],
        {4915200, 16, 800, 1600, 400},
        10,
        10,
        binding_reports={
            'reference': reference,
            'candidate': binding_report('std', direct_output=False),
        },
    )

    assert result['status'] == 'PASS'
    assert result['binding_report_complete'] is True


def test_equal_negative_native_output_pointers_are_inconclusive_in_c_vs_d():
    """A negative address cannot prove the Config C direct-output identity."""
    reference = binding_report()
    reference['outputs'][0]['output_pointer'] = -16384
    reference['outputs'][0]['ort_pointer'] = -16384
    result = TRACE_COMPARE.compare(
        [kernel('inference_kernel')],
        [kernel('inference_kernel')],
        {4915200, 16, 800, 1600, 400},
        10,
        10,
        binding_reports={
            'reference': reference,
            'candidate': binding_report('std', direct_output=False),
        },
    )

    assert result['status'] == 'INCONCLUSIVE'
    assert result['binding_report_complete'] is False
