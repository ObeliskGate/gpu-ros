# Copyright 2026 Maintainer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

"""Tests for the strict ROCprofiler JSON loader and AMD evidence rules."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = PACKAGE_ROOT / 'scripts' / 'analyze_rocprof_traces.py'
SPEC = importlib.util.spec_from_file_location('analyze_rocprof_traces', SCRIPT_PATH)
ROC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROC)

FIXTURE_ROOT = Path(__file__).parent / 'fixtures'
MANIFEST = FIXTURE_ROOT / 'rocprofv3_capture_manifest.json'


def memory_copy(operation, size, source='host', destination='device'):
    return {
        'kind': 'memory_copy',
        'operation': operation,
        'direction': operation,
        'bytes': size,
        'src_agent': source,
        'dst_agent': destination,
    }


def kernel(name):
    return {'kind': 'kernel_dispatch', 'kernel_name': name, 'operation': name}


def test_fixture_metadata_is_not_an_activity_event():
    capture = ROC.load_capture([FIXTURE_ROOT / 'rocprofv3_minimal.json'])

    assert len(capture.events) == 4
    assert len(capture.memory_copy_events) == 2
    assert len(capture.kernel_events) == 2
    assert all(event['rocprof_domain'] in {'memory_copy', 'kernel_dispatch'}
               for event in capture.events)
    assert capture.sections['memory_copy_trace']['record_count'] == 2
    assert capture.sections['kernel_trace']['record_count'] == 2


def test_operation_lookup_is_metadata_driven_and_numeric_enum_is_only_sanity_check():
    capture = ROC.load_capture([FIXTURE_ROOT / 'rocprofv3_minimal.json'])
    summary = ROC.summarize(capture.events, {16, 8})

    assert summary['memory_totals'] == {
        'H2D': {'count': 1, 'bytes': 16},
        'D2H': {'count': 1, 'bytes': 8},
    }
    assert not capture.diagnostics

    document = json.loads((FIXTURE_ROOT / 'rocprofv3_minimal.json').read_text())
    # Keep numeric enum 2 but make the metadata lookup claim enum 3.
    document['rocprofiler-sdk-tool'][0]['strings']['buffer_records'][0]['operations'][2] = \
        'MEMORY_COPY_DEVICE_TO_HOST'
    changed = FIXTURE_ROOT / 'rocprofv3_enum_mismatch.json'
    changed.write_text(json.dumps(document), encoding='utf-8')
    try:
        mismatch = ROC.load_capture([changed])
    finally:
        changed.unlink()
    assert any(item['kind'] == 'schema_version_mismatch'
               for item in mismatch.diagnostics)
    # Metadata lookup says D2H even though the numeric value is inconsistent.  The
    # loader must not silently replace the metadata fact with a magic-number
    # guess.
    assert mismatch.memory_copy_events[0]['direction'] == 'D2H'


def test_missing_operation_lookup_does_not_guess_numeric_direction(tmp_path):
    document = json.loads((FIXTURE_ROOT / 'rocprofv3_minimal.json').read_text())
    document['rocprofiler-sdk-tool'][0]['strings']['buffer_records'][0]['operations'] = []
    path = tmp_path / 'unknown-operation.json'
    path.write_text(json.dumps(document), encoding='utf-8')

    capture = ROC.load_capture([path])
    event = capture.memory_copy_events[0]
    assert event['direction'] == 'unknown'
    assert any(item['kind'] == 'operation_lookup_failed'
               for item in capture.diagnostics)


def test_agent_handles_are_process_local(tmp_path):
    first = json.loads((FIXTURE_ROOT / 'rocprofv3_minimal.json').read_text())
    second = copy.deepcopy(first)
    process = second['rocprofiler-sdk-tool'][0]
    process['metadata']['pid'] = 202
    # Reuse the same numeric handles, but invert the process-local table.  The
    # operation metadata remains H2D, while source/destination roles must be
    # decoded from this process's own agent table.
    process['agents'][0], process['agents'][1] = process['agents'][1], process['agents'][0]
    process['agents'][0]['type'] = 2
    process['agents'][0]['vendor_name'] = 'AMD'
    process['agents'][1]['type'] = 1
    process['agents'][1]['vendor_name'] = 'CPU'
    path = tmp_path / 'second-process.json'
    path.write_text(json.dumps(second), encoding='utf-8')
    capture = ROC.load_capture([FIXTURE_ROOT / 'rocprofv3_minimal.json', path])
    assert len(capture.memory_copy_events) == 4
    assert {event['source_agent_handle'] for event in capture.memory_copy_events} == {10, 11}
    assert all(event['direction'] == direction for event, direction in zip(
        capture.memory_copy_events, ('H2D', 'D2H', 'H2D', 'D2H')))


def test_kernel_dispatch_maps_through_kernel_symbols_without_complete_filtering():
    capture = ROC.load_capture([FIXTURE_ROOT / 'rocprofv3_minimal.json'])
    names = [event['kernel_name'] for event in capture.kernel_events]
    assert names == ['provider_kernel.kd', '__amd_rocclr_copyBuffer.kd']
    assert all(event['kernel_operation_name'] == 'KERNEL_DISPATCH_COMPLETE'
               for event in capture.kernel_events)


def test_malformed_json_and_schema_are_tooling_errors(tmp_path):
    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{not-json', encoding='utf-8')
    with pytest.raises(ROC.RocprofJsonError):
        ROC.load_capture([malformed])

    missing_wrapper = tmp_path / 'missing-wrapper.json'
    missing_wrapper.write_text('{}', encoding='utf-8')
    with pytest.raises(ROC.RocprofJsonError):
        ROC.load_capture([missing_wrapper])

    wrong_section = json.loads((FIXTURE_ROOT / 'rocprofv3_minimal.json').read_text())
    wrong_section['rocprofiler-sdk-tool'][0]['buffer_records']['memory_copy'] = {}
    wrong_path = tmp_path / 'wrong-section.json'
    wrong_path.write_text(json.dumps(wrong_section), encoding='utf-8')
    with pytest.raises(ROC.RocprofJsonError):
        ROC.load_capture([wrong_path])


def test_cli_returns_tooling_error_exit_code_for_malformed_capture(tmp_path):
    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{not-json', encoding='utf-8')
    output = tmp_path / 'report.json'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            '--self-report',
            '--trace',
            str(malformed),
            '--output-json',
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert 'ERROR:' in completed.stderr
    assert not output.exists()


def test_malformed_records_are_retained_as_inconclusive_evidence(tmp_path):
    document = json.loads((FIXTURE_ROOT / 'rocprofv3_minimal.json').read_text())
    process = document['rocprofiler-sdk-tool'][0]
    process['buffer_records']['memory_copy'].append({
        'kind': 10,
        'operation': 99,
        'src_agent_id': {'handle': 999},
        'dst_agent_id': {'handle': 11},
    })
    process['buffer_records']['kernel_dispatch'].append({
        'kind': 11,
        'operation': 2,
        'dispatch_info': {'kernel_id': 999, 'agent_id': {'handle': 11}},
    })
    path = tmp_path / 'record-errors.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    capture = ROC.load_capture([path])

    assert len(capture.memory_copy_events) == 3
    assert len(capture.kernel_events) == 3
    assert capture.diagnostics
    result = ROC.compare(
        capture.events, capture.events, {16, 8},
        std_frames=1, managed_frames=1,
        std_capture=capture, managed_capture=capture,
        std_manifest=json.loads(MANIFEST.read_text()),
        managed_manifest=json.loads(MANIFEST.read_text()),
    )
    assert result['status'] == 'INCONCLUSIVE'
    assert result['capture_completeness']['std']['parser_diagnostic_count'] > 0


def test_requested_empty_memory_copy_domain_is_inconclusive():
    capture = ROC.load_capture([FIXTURE_ROOT / 'rocprofv3_empty_memory_copy.json'])
    manifest = json.loads(MANIFEST.read_text())
    details = ROC.capture_completeness(capture, manifest)
    assert details['memory_copy']['status'] == 'copy_domain_incomplete'
    assert details['kernel']['status'] == 'complete'

    result = ROC.compare(
        capture.events, capture.events, set(),
        std_frames=1, managed_frames=1,
        require_adapter_directions=True,
        std_capture=capture, managed_capture=capture,
        std_manifest=manifest, managed_manifest=manifest,
    )
    assert result['status'] == 'INCONCLUSIVE'
    assert result['criteria']['memory_copy_manifest_complete'] is False


def test_staging_shaped_h2d_d2h_evidence_is_not_claimed_as_adapter_proof():
    result = ROC.compare(
        [kernel('std_provider_kernel')],
        [memory_copy('hipMemcpyHtoD', 16), memory_copy('hipMemcpyDtoH', 8, 'device', 'host')],
        {16, 8}, std_frames=10, managed_frames=10,
        adapter_directions=('H2D', 'D2H'), require_adapter_directions=False,
    )

    assert result['status'] == 'PASS'
    assert result['memory_copy_failures'] == []
    assert {item['classification'] for item in result['adapter_copy_evidence']} == {
        'staging_shaped_evidence'}
    assert 'prove' in result['staging_policy']['description']


def test_managed_only_tensor_sized_d2d_copy_is_fail():
    result = ROC.compare(
        [kernel('provider_kernel')],
        [kernel('provider_kernel'), memory_copy('hipMemcpyDtoD', 16, 'device', 'device')],
        {16}, std_frames=10, managed_frames=10,
    )
    assert result['status'] == 'FAIL'
    assert result['memory_copy_failures'][0]['reason'].startswith(
        'managed-only tensor-sized')


def test_missing_expected_d2h_is_inconclusive_not_zero_copy_pass():
    result = ROC.compare(
        [memory_copy('hipMemcpyHtoD', 16)],
        [memory_copy('hipMemcpyHtoD', 16)],
        {16}, std_frames=10, managed_frames=10,
        adapter_directions=('H2D', 'D2H'), require_adapter_directions=True,
    )
    assert result['status'] == 'INCONCLUSIVE'
    assert result['criteria']['expected_adapter_directions_observed'] is False
    assert result['memory_copy_failures'] == []


def test_ordinary_kernel_count_difference_is_diagnostic_only():
    result = ROC.compare(
        [kernel('provider_kernel')],
        [kernel('provider_kernel'), kernel('provider_kernel')],
        set(), std_frames=10, managed_frames=10,
    )
    assert result['status'] == 'PASS'
    assert result['unresolved_kernel_evidence'] == []
    assert result['kernel_evidence']['count_deltas'][0]['count_delta'] == 1


def test_shared_copybuffer_delta_is_amd_unresolved_evidence_not_generic_failure():
    result = ROC.compare(
        [kernel('__amd_rocclr_copyBuffer.kd')],
        [kernel('__amd_rocclr_copyBuffer.kd'), kernel('__amd_rocclr_copyBuffer.kd')],
        set(), std_frames=10, managed_frames=10,
    )
    assert result['status'] == 'INCONCLUSIVE'
    assert result['memory_copy_failures'] == []
    assert result['kernel_evidence']['copyBuffer'][0]['count_delta'] == 1
    assert result['unresolved_kernel_evidence'][0]['classification'] == \
        'unresolved_kernel_evidence'


def test_common_report_does_not_turn_shared_copybuffer_delta_into_failure():
    from pathlib import Path as _Path
    common_path = _Path(__file__).parents[1] / 'scripts' / 'copy_audit_common.py'
    common_spec = importlib.util.spec_from_file_location('copy_audit_common_history', common_path)
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    result = common.build_pair_report(
        [kernel('__amd_rocclr_copyBuffer.kd')],
        [kernel('__amd_rocclr_copyBuffer.kd'), kernel('__amd_rocclr_copyBuffer.kd')],
        set(), reference_frames=10, managed_frames=10,
    )
    assert result['status'] == 'PASS'


@pytest.mark.skipif(
    os.environ.get('ROCPROF_RUN_FULL_TRACE_REGRESSION') != '1',
    reason='set ROCPROF_RUN_FULL_TRACE_REGRESSION=1 for the local 283 MB trace regression',
)
def test_full_mi3501x_rtdetr_trace_regression():
    root = Path(__file__).parents[3]
    sample_root = root / 'inner_docs' / 'rocprof_samples' / \
        'rtdetr_amd_transport_audit_mi3501x_20260808_081138'
    std_path = sample_root / 'std' / 'rtdetr_managed_attach_results.json'
    managed_path = sample_root / 'managed' / 'rtdetr_managed_attach_results.json'
    if not std_path.exists() or not managed_path.exists():
        pytest.skip('local raw MI3501X traces are not available')
    std_capture = ROC.load_capture([std_path])
    managed_capture = ROC.load_capture([managed_path])
    manifest = json.loads(MANIFEST.read_text())
    result = ROC.compare(
        std_capture.events, managed_capture.events,
        {4915200, 800, 1600, 400},
        std_frames=391, managed_frames=393,
        adapter_directions=('H2D', 'D2H'), require_adapter_directions=True,
        std_capture=std_capture, managed_capture=managed_capture,
        std_manifest=manifest, managed_manifest=manifest,
    )
    assert std_capture.sections['memory_copy_trace']['record_count'] == 386
    assert managed_capture.sections['memory_copy_trace']['record_count'] == 393
    assert std_capture.sections['kernel_trace']['record_count'] == 313432
    assert managed_capture.sections['kernel_trace']['record_count'] == 320688
    assert std_capture.memory_copy_events[0]['direction'] == 'H2D'
    assert {event['direction'] for event in std_capture.memory_copy_events} == {'H2D'}
    assert {event['direction'] for event in managed_capture.memory_copy_events} == {'H2D'}
    assert result['memory_copy_failures'] == []
    assert result['status'] == 'INCONCLUSIVE'
    assert result['capture_completeness']['managed']['memory_copy']['status'] == 'complete'
    std_copybuffer = sum(item['std_count'] for item in result['kernel_evidence']['copyBuffer'])
    managed_copybuffer = sum(
        item['managed_count'] for item in result['kernel_evidence']['copyBuffer'])
    assert std_copybuffer == 2316
    assert managed_copybuffer == 3930
    assert result['unresolved_kernel_evidence']
