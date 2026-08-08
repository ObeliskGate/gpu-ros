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

"""Compare Config C and Managed Nsight Systems CUDA trace exports.

The historical command line and JSON keys are retained.  The closure decision
now uses explicit memory-copy records as the primary copy evidence.  Kernel
set/count differences remain in the report as diagnostics; a copy-looking
Managed-only kernel produces ``INCONCLUSIVE`` until its payload role is
resolved, rather than an automatic ``FAIL``.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from copy_audit_common import (
        build_pair_report,
        load_trace_records,
        self_report_from_summary,
        summarize_events,
    )
    from copy_audit_common import (
        byte_count as _byte_count,
    )
except ModuleNotFoundError:  # Direct source-tree imports used by pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copy_audit_common import (  # type: ignore[no-redef]
        build_pair_report,
        load_trace_records,
        self_report_from_summary,
        summarize_events,
    )
    from copy_audit_common import (
        byte_count as _byte_count,
    )


def parse_args():
    """Parse paired or self-report command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Config C and Managed nsys cuda_gpu_trace JSON files.')
    parser.add_argument('--config-c-trace', type=Path)
    parser.add_argument('--managed-trace', type=Path)
    parser.add_argument(
        '--trace', action='append', type=Path,
        help='Trace JSON for --self-report; may be repeated for multi-process output')
    parser.add_argument('--lane', default='unknown')
    parser.add_argument('--self-report', action='store_true')
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--config-c-frames', type=int)
    parser.add_argument('--managed-frames', type=int)
    parser.add_argument(
        '--max-payload-copy-rate-delta',
        default=0.05,
        type=float,
        help='Maximum allowed M-minus-C copies per frame for a payload signature')
    parser.add_argument(
        '--payload-size',
        action='append',
        default=[],
        type=int,
        help='Expected payload size in bytes; may be repeated')
    parser.add_argument(
        '--boundary-evidence', type=Path,
        help='Optional JSON evidence confirming or leaving a boundary copy unresolved')
    parser.add_argument('--config-c-binding-report', type=Path)
    parser.add_argument('--managed-binding-report', type=Path)
    parser.add_argument(
        '--kernel-payload-risk', action='append', default=[],
        help='Managed-only kernel name that cannot be ruled out as payload movement')
    return parser.parse_args()


def load_events(path):
    """Load one nsys JSON report, preserving the historical helper API."""
    return list(load_trace_records([Path(path)]))


def event_field(event, name, default=''):
    """Read a field while tolerating formatter capitalization changes."""
    if name in event:
        return event[name]
    lower_name = name.lower()
    for key, value in event.items():
        if str(key).lower() == lower_name:
            return value
    return default


def byte_count(event):
    """Preserve the historical byte-count helper exported by this module."""
    return _byte_count(event)


def summarize(events, payload_sizes):
    """Summarize stable kernel names and memory-copy signatures.

    This wrapper keeps the old Python API used by downstream audit notebooks.
    New consumers should use ``copy_audit_common.summarize_events`` directly.
    """
    return summarize_events(events, payload_sizes, platform='nsys')


def compare(
        config_c_events,
        managed_events,
        payload_sizes,
        config_c_frames=None,
        managed_frames=None,
        max_payload_copy_rate_delta=0.05,
        boundary_evidence=None,
        profiler_complete=True,
        kernel_payload_risk_names=None,
        binding_reports=None):
    """Build the machine-readable transport audit comparison.

    ``config_c_events`` and ``managed_events`` are raw Nsight rows.  The
    resulting report keeps the historical ``pass``/``criteria``/``memcpy``
    sections and adds explicit PASS/FAIL/INCONCLUSIVE evidence sections.
    """
    result = build_pair_report(
        config_c_events,
        managed_events,
        payload_sizes,
        reference_frames=config_c_frames,
        managed_frames=managed_frames,
        max_payload_copy_rate_delta=max_payload_copy_rate_delta,
        platform='nsys',
        reference_lane='config_c',
        boundary_evidence=boundary_evidence,
        profiler_complete=profiler_complete,
        kernel_payload_risk_names=kernel_payload_risk_names,
        binding_reports=binding_reports,
    )
    return result


def load_boundary_evidence(path):
    if path is None:
        return None
    with path.open(encoding='utf-8') as evidence_file:
        evidence = json.load(evidence_file)
    if not isinstance(evidence, dict):
        raise TypeError(f'{path}: boundary evidence must be a JSON object')
    return evidence


def load_binding_reports(config_c_path, managed_path):
    """Load the first-frame reports used to complete pointer/lifetime evidence."""
    if config_c_path is None and managed_path is None:
        return None
    reports = {}
    for lane, path in (('reference', config_c_path), ('managed', managed_path)):
        if path is None:
            reports[lane] = None
            continue
        with path.open(encoding='utf-8') as report_file:
            reports[lane] = json.load(report_file)
    return reports


def self_report(events, lane, payload_sizes, frames=None, profiler_complete=True):
    """Return one lane's machine-readable self report."""
    summary = summarize_events(events, payload_sizes, platform='nsys', frame_count=frames)
    return self_report_from_summary(summary, lane, 'nsys', frames, profiler_complete)


def main():
    """Run the CUDA trace comparison or a single-lane report."""
    args = parse_args()
    try:
        if args.self_report:
            if not args.trace:
                raise ValueError('--self-report requires at least one --trace')
            if args.config_c_trace or args.managed_trace:
                raise ValueError('--self-report cannot be combined with paired trace arguments')
            events = load_trace_records(args.trace)
            result = self_report(
                events, args.lane, set(args.payload_size),
                frames=args.config_c_frames or args.managed_frames,
            )
            result['profiler_complete'] = True
        else:
            if not args.config_c_trace or not args.managed_trace:
                raise ValueError(
                    'paired comparison requires --config-c-trace and --managed-trace')
            evidence = load_boundary_evidence(args.boundary_evidence)
            result = compare(
                load_events(args.config_c_trace),
                load_events(args.managed_trace),
                set(args.payload_size),
                args.config_c_frames,
                args.managed_frames,
                args.max_payload_copy_rate_delta,
                boundary_evidence=evidence,
                kernel_payload_risk_names=args.kernel_payload_risk,
                binding_reports=load_binding_reports(
                    args.config_c_binding_report, args.managed_binding_report),
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if args.self_report:
        print(
            f"{result['lane']} memory-copy events: "
            f"{sum(item['count'] for item in result['memory_copy'])}")
        print('SELF REPORT')
        return 0

    print(f"Status: {result['status']}")
    print(
        'Kernel name sets match (diagnostic): '
        f"{result['criteria']['kernel_name_sets_match']}")
    print(
        'Managed has no extra memcpy signature: '
        f"{result['criteria']['managed_has_no_extra_memcpy_signature']}")
    print(
        'Managed has no extra payload copy rate: '
        f"{result['criteria']['managed_has_no_extra_payload_copy_rate']}")
    print(f"Managed boundary status: {result['status']}")
    for delta in result['memcpy']['memory_total_deltas']:
        print(
            f"{delta['direction']}: count delta={delta['count_delta']}, "
            f"byte delta={delta['byte_delta']}")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
