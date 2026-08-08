#!/usr/bin/env python3
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

"""
Compare explicit CUDA copy records for two fixed-input NVIDIA lanes.

This report is deliberately a Config C versus Config D diagnostic.  A
positive D-minus-C H2D or D2H delta is evidence that the standard ROS 2 path
adds host/device transfers, but this command is not a Managed zero-copy
closure verdict.  Kernel name differences remain diagnostic and are never
classified as memory copies without an explicit copy record.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

try:
    from copy_audit_common import (
        counter_deltas,
        load_trace_records,
        looks_like_payload_kernel,
        self_report_from_summary,
        serialize_counter,
        summarize_events,
    )
except ModuleNotFoundError:  # Direct source-tree imports used by pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copy_audit_common import (  # type: ignore[no-redef]
        counter_deltas,
        load_trace_records,
        looks_like_payload_kernel,
        self_report_from_summary,
        serialize_counter,
        summarize_events,
    )


DIRECTIONS = ('H2D', 'D2H', 'D2D', 'H2H', 'unknown')


def parse_args() -> argparse.Namespace:
    """Parse the C-vs-D comparison arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Config C and Config D NVIDIA CUDA copy traces.')
    parser.add_argument(
        '--reference-trace', action='append', required=True, type=Path,
        help='Config C nsys cuda_gpu_trace JSON; may be repeated')
    parser.add_argument(
        '--candidate-trace', action='append', required=True, type=Path,
        help='Config D nsys cuda_gpu_trace JSON; may be repeated')
    parser.add_argument('--reference-label', default='config_c')
    parser.add_argument('--candidate-label', default='config_d')
    parser.add_argument('--reference-frames', required=True, type=int)
    parser.add_argument('--candidate-frames', required=True, type=int)
    parser.add_argument('--reference-binding-report', type=Path)
    parser.add_argument('--candidate-binding-report', type=Path)
    parser.add_argument(
        '--payload-size', action='append', default=[], type=int,
        help='Additional tensor size in bytes; binding reports are included automatically')
    parser.add_argument('--output-json', required=True, type=Path)
    return parser.parse_args()


def _load_binding_report(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load one first-frame binding report, if supplied."""
    if path is None:
        return None
    with path.open(encoding='utf-8') as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise TypeError(f'{path}: binding report must be a JSON object')
    return report


def _binding_errors(report: Optional[Mapping[str, Any]], label: str) -> list[str]:
    """Return missing first-frame fields without hiding copy evidence."""
    if report is None:
        return [f'{label}: binding report was not supplied']
    errors = []
    if report.get('first_frame') is not True:
        errors.append(f'{label}: first_frame marker missing')
    for side in ('inputs', 'outputs'):
        records = report.get(side)
        if not isinstance(records, list) or not records:
            errors.append(f'{label}: {side} records missing')
            continue
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                errors.append(f'{label}: {side}[{index}] is not an object')
                continue
            for key in ('name', 'bytes', 'storage', 'lifetime_path'):
                if key not in record:
                    errors.append(f'{label}: {side}[{index}] missing {key}')
            if not record.get('lifetime_path'):
                errors.append(f'{label}: {side}[{index}] lifetime path empty')
    return errors


def _binding_payload_sizes(report: Optional[Mapping[str, Any]]) -> Set[int]:
    """Extract exact input/output tensor byte sizes from a binding report."""
    sizes: Set[int] = set()
    if report is None:
        return sizes
    for side in ('inputs', 'outputs'):
        records = report.get(side, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            value = record.get('bytes')
            if isinstance(value, bool):
                continue
            try:
                size = int(value)
            except (TypeError, ValueError):
                continue
            if size > 0:
                sizes.add(size)
    return sizes


def _load_events(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    """Load and flatten one or more nsys JSON exports."""
    return list(load_trace_records(paths))


def _totals(summary: Mapping[str, Any], direction: str) -> Dict[str, int]:
    """Return a zero-filled count/byte pair for one direction."""
    value = summary['memory_totals'].get(direction, {'count': 0, 'bytes': 0})
    return {'count': int(value.get('count', 0)), 'bytes': int(value.get('bytes', 0))}


def _memory_total_deltas(
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        reference_frames: int,
        candidate_frames: int) -> list[Dict[str, Any]]:
    """Return absolute and per-frame candidate-minus-reference totals."""
    rows = []
    for direction in DIRECTIONS:
        reference_data = _totals(reference, direction)
        candidate_data = _totals(candidate, direction)
        count_delta = candidate_data['count'] - reference_data['count']
        byte_delta = candidate_data['bytes'] - reference_data['bytes']
        rows.append({
            'direction': direction,
            'reference': reference_data,
            'candidate': candidate_data,
            'count_delta': count_delta,
            'byte_delta': byte_delta,
            'reference_count_per_frame': reference_data['count'] / reference_frames,
            'candidate_count_per_frame': candidate_data['count'] / candidate_frames,
            'count_delta_per_frame': (
                candidate_data['count'] / candidate_frames -
                reference_data['count'] / reference_frames),
            'reference_bytes_per_frame': reference_data['bytes'] / reference_frames,
            'candidate_bytes_per_frame': candidate_data['bytes'] / candidate_frames,
            'byte_delta_per_frame': (
                candidate_data['bytes'] / candidate_frames -
                reference_data['bytes'] / reference_frames),
            'candidate_more': count_delta > 0 or byte_delta > 0,
        })
    return rows


def _payload_deltas(
        reference: Mapping[str, Any], candidate: Mapping[str, Any],
        payload_sizes: Iterable[int], reference_frames: int,
        candidate_frames: int) -> list[Dict[str, Any]]:
    """Return per-size and per-direction copy deltas for known tensor sizes."""
    reference_counts = reference['payload_copy_counts']
    candidate_counts = candidate['payload_copy_counts']
    rows = []
    for size in sorted(set(payload_sizes)):
        for direction in DIRECTIONS:
            key = (size, direction)
            reference_count = int(reference_counts.get(key, 0))
            candidate_count = int(candidate_counts.get(key, 0))
            if reference_count == 0 and candidate_count == 0:
                continue
            rows.append({
                'size_bytes': size,
                'direction': direction,
                'reference_count': reference_count,
                'candidate_count': candidate_count,
                'count_delta': candidate_count - reference_count,
                'reference_per_frame': reference_count / reference_frames,
                'candidate_per_frame': candidate_count / candidate_frames,
                'delta_per_frame': (
                    candidate_count / candidate_frames -
                    reference_count / reference_frames),
            })
    return rows


def _candidate_only_payload_evidence(
        reference: Mapping[str, Any], candidate: Mapping[str, Any],
        payload_sizes: Set[int]) -> list[Dict[str, Any]]:
    """Identify candidate-only or candidate-increased tensor-sized signatures."""
    rows = []
    for signature, candidate_count in candidate['memcopies'].items():
        size = signature[1]
        if size not in payload_sizes:
            continue
        reference_count = int(reference['memcopies'].get(signature, 0))
        if candidate_count <= reference_count:
            continue
        rows.append({
            'signature': list(signature),
            'reference_count': reference_count,
            'candidate_count': candidate_count,
            'count_delta': candidate_count - reference_count,
            'classification': 'candidate_only_or_increased_tensor_sized_copy',
        })
    return rows


def compare(
        reference_events: Iterable[Mapping[str, Any]],
        candidate_events: Iterable[Mapping[str, Any]],
        payload_sizes: Iterable[int],
        reference_frames: int,
        candidate_frames: int,
        reference_label: str = 'config_c',
        candidate_label: str = 'config_d',
        binding_reports: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a lane-neutral C-vs-D copy delta report."""
    if reference_frames <= 0 or candidate_frames <= 0:
        raise ValueError('frame counts must be positive')
    payload_set = {int(size) for size in payload_sizes if int(size) > 0}
    reference_summary = summarize_events(
        reference_events, payload_set, platform='nsys', frame_count=reference_frames)
    candidate_summary = summarize_events(
        candidate_events, payload_set, platform='nsys', frame_count=candidate_frames)
    trace_complete = (
        reference_summary['kernel_event_count'] > 0 or
        bool(reference_summary['memcopies'])) and (
            candidate_summary['kernel_event_count'] > 0 or
            bool(candidate_summary['memcopies']))

    reports = binding_reports or {}
    reference_binding = reports.get('reference')
    candidate_binding = reports.get('candidate')
    binding_errors = (
        _binding_errors(reference_binding, reference_label) +
        _binding_errors(candidate_binding, candidate_label))
    deltas = _memory_total_deltas(
        reference_summary, candidate_summary, reference_frames, candidate_frames)
    host_device_directions = {'H2D', 'D2H'}
    positive_host_device = [
        row for row in deltas
        if row['direction'] in host_device_directions and row['candidate_more']
    ]
    reference_kernel_names = set(reference_summary['kernels'])
    candidate_kernel_names = set(candidate_summary['kernels'])
    kernel_only_differences = [
        {
            'name': name,
            'reference_count': reference_summary['kernels'][name],
            'candidate_count': candidate_summary['kernels'][name],
            'classification': 'diagnostic_kernel_set_difference',
        }
        for name in sorted(candidate_kernel_names - reference_kernel_names)
    ]
    unresolved_payload_copy_risk = [
        {
            'name': row['name'],
            'reason': (
                'Config D-only kernel could carry tensor payload; kernel trace '
                'cannot prove otherwise'),
        }
        for row in kernel_only_differences
        if looks_like_payload_kernel(row['name'])
    ]
    evidence_status = 'PASS' if trace_complete and not binding_errors else 'INCONCLUSIVE'
    return {
        'schema_version': 1,
        'report_type': 'nvidia_lane_copy_comparison',
        'audit_scope': 'config_c_vs_config_d_fixed_input',
        'status': evidence_status,
        'final_status': evidence_status,
        'zero_copy_status': 'NOT_APPLICABLE',
        'status_note': (
            'PASS means both lane traces and binding reports are complete; it does '
            'not mean Config D is zero-copy.'),
        'reference_lane': reference_label,
        'candidate_lane': candidate_label,
        'reference_frames': reference_frames,
        'candidate_frames': candidate_frames,
        'trace_complete': trace_complete,
        'binding_report_complete': not binding_errors,
        'binding_report_errors': binding_errors,
        'payload_sizes_bytes': sorted(payload_set),
        'binding_payload_sizes': {
            reference_label: sorted(_binding_payload_sizes(reference_binding)),
            candidate_label: sorted(_binding_payload_sizes(candidate_binding)),
        },
        'self': {
            reference_label: self_report_from_summary(
                reference_summary, reference_label, 'nsys', reference_frames, trace_complete),
            candidate_label: self_report_from_summary(
                candidate_summary, candidate_label, 'nsys', candidate_frames, trace_complete),
        },
        'memory_totals': {
            reference_label: reference_summary['memory_totals'],
            candidate_label: candidate_summary['memory_totals'],
        },
        'memory_total_deltas': deltas,
        'host_device_copy_evidence': {
            'candidate_more_h2d_or_d2h': bool(positive_host_device),
            'positive_deltas': positive_host_device,
            'interpretation': (
                'Positive D-minus-C H2D/D2H count or byte deltas are direct evidence '
                'of additional explicit host/device copy records in D.'),
        },
        'memory_copy_signature_deltas': counter_deltas(
            reference_summary['memcopies'], candidate_summary['memcopies']),
        'payload_copy_deltas': _payload_deltas(
            reference_summary, candidate_summary, payload_set,
            reference_frames, candidate_frames),
        'boundary_payload_copy_evidence': _candidate_only_payload_evidence(
            reference_summary, candidate_summary, payload_set),
        'memory_copy_failures': [],
        'kernel_only_differences': kernel_only_differences,
        'unresolved_payload_copy_risk': unresolved_payload_copy_risk,
        'kernel_names': {
            reference_label: serialize_counter(reference_summary['kernels']),
            candidate_label: serialize_counter(candidate_summary['kernels']),
            'missing_from_candidate': sorted(reference_kernel_names - candidate_kernel_names),
            'extra_in_candidate': sorted(candidate_kernel_names - reference_kernel_names),
            'event_count_deltas': counter_deltas(
                reference_summary['kernels'], candidate_summary['kernels']),
        },
        'memory_copy': {
            reference_label: serialize_counter(reference_summary['memcopies']),
            candidate_label: serialize_counter(candidate_summary['memcopies']),
        },
        'binding_reports': {
            reference_label: reference_binding,
            candidate_label: candidate_binding,
        },
    }


def main() -> int:
    """Run the C-vs-D comparison and write a machine-readable report."""
    args = parse_args()
    try:
        reference_binding = _load_binding_report(args.reference_binding_report)
        candidate_binding = _load_binding_report(args.candidate_binding_report)
        binding_sizes = (
            _binding_payload_sizes(reference_binding) |
            _binding_payload_sizes(candidate_binding))
        payload_sizes = binding_sizes | {int(size) for size in args.payload_size}
        result = compare(
            _load_events(args.reference_trace),
            _load_events(args.candidate_trace),
            payload_sizes,
            args.reference_frames,
            args.candidate_frames,
            args.reference_label,
            args.candidate_label,
            {'reference': reference_binding, 'candidate': candidate_binding},
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(
        f'{args.reference_label} frames={args.reference_frames}; '
        f'{args.candidate_label} frames={args.candidate_frames}')
    print('Explicit memory-copy delta (candidate minus reference):')
    for row in result['memory_total_deltas']:
        print(
            f"  {row['direction']}: count {row['reference']['count']} -> "
            f"{row['candidate']['count']} (delta {row['count_delta']}, "
            f"per-frame {row['count_delta_per_frame']:.6f}); bytes "
            f"{row['reference']['bytes']} -> {row['candidate']['bytes']} "
            f"(delta {row['byte_delta']}, "
            f"per-frame {row['byte_delta_per_frame']:.2f})")
    if result['host_device_copy_evidence']['candidate_more_h2d_or_d2h']:
        print('RESULT: Config D has additional explicit H2D/D2H copy evidence.')
    else:
        print('RESULT: no positive explicit H2D/D2H delta was observed.')
    print(f"Evidence status: {result['status']}")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
