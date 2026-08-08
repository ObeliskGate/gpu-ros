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

"""Parse ROCprofiler-v3 JSON and audit the AMD Managed inference boundary."""

import argparse
import json
from pathlib import Path
import sys

try:
    from copy_audit_common import (
        build_pair_report,
        load_trace_records,
        self_report_from_summary,
        summarize_events,
    )
except ModuleNotFoundError:  # Direct source-tree imports used by pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copy_audit_common import (  # type: ignore[no-redef]
        build_pair_report,
        load_trace_records,
        self_report_from_summary,
        summarize_events,
    )


def parse_args():
    """Parse paired or self-report arguments."""
    parser = argparse.ArgumentParser(
        description='Normalize ROCprofiler-v3 memory-copy and kernel JSON traces.')
    parser.add_argument('--std-trace', action='append', type=Path)
    parser.add_argument('--managed-trace', action='append', type=Path)
    parser.add_argument('--trace', action='append', type=Path)
    parser.add_argument('--self-report', action='store_true')
    parser.add_argument('--lane', default='unknown')
    parser.add_argument('--std-frames', type=int)
    parser.add_argument('--managed-frames', type=int)
    parser.add_argument('--frame-count', type=int)
    parser.add_argument('--payload-size', action='append', default=[], type=int)
    parser.add_argument(
        '--expected-adapter-direction', action='append', default=[],
        choices=['H2D', 'D2H', 'D2D', 'H2H', 'unknown'],
        help='Managed-only copies in this direction are explicit adapter evidence')
    parser.add_argument('--boundary-evidence', type=Path)
    parser.add_argument('--std-binding-report', type=Path)
    parser.add_argument('--managed-binding-report', type=Path)
    parser.add_argument(
        '--kernel-payload-risk', action='append', default=[],
        help='Managed-only kernel name that cannot be ruled out as payload movement')
    parser.add_argument('--output-json', required=True, type=Path)
    return parser.parse_args()


def load_events(paths):
    """Load one or more ROCprofiler JSON files, including PID fragments."""
    return list(load_trace_records([Path(path) for path in paths]))


def summarize(events, payload_sizes, frames=None):
    """Public parser helper used by tests and offline audit notebooks."""
    return summarize_events(
        events, payload_sizes, platform='rocprof', frame_count=frames)


def load_boundary_evidence(path):
    if path is None:
        return None
    with path.open(encoding='utf-8') as evidence_file:
        evidence = json.load(evidence_file)
    if not isinstance(evidence, dict):
        raise TypeError(f'{path}: boundary evidence must be a JSON object')
    return evidence


def load_binding_reports(std_path, managed_path):
    """Load first-frame pointer/lifetime reports for both AMD lanes."""
    if std_path is None and managed_path is None:
        return None
    reports = {}
    for lane, path in (('reference', std_path), ('managed', managed_path)):
        if path is None:
            reports[lane] = None
            continue
        with path.open(encoding='utf-8') as report_file:
            reports[lane] = json.load(report_file)
    return reports


def compare(
        std_events,
        managed_events,
        payload_sizes,
        std_frames=None,
        managed_frames=None,
        adapter_directions=('H2D', 'D2H'),
        boundary_evidence=None,
        profiler_complete=True,
        kernel_payload_risk_names=None,
        binding_reports=None):
    """Build an AMD std-vs-Managed report with explicit staging evidence."""
    result = build_pair_report(
        std_events,
        managed_events,
        payload_sizes,
        reference_frames=std_frames,
        managed_frames=managed_frames,
        platform='rocprof',
        reference_lane='std',
        managed_adapter_directions=adapter_directions,
        boundary_evidence=boundary_evidence,
        profiler_complete=profiler_complete,
        kernel_payload_risk_names=kernel_payload_risk_names,
        binding_reports=binding_reports,
    )
    result['platform'] = 'rocprof'
    result['staging_policy'] = {
        'managed_adapter_directions': sorted(set(adapter_directions)),
        'description': (
            'Managed-only H2D/D2H records are explicit std-to-device and '
            'device-to-std adapter evidence; the inference boundary is audited '
            'for additional tensor-sized copies.'
        ),
    }
    return result


def self_report(events, lane, payload_sizes, frames=None, profiler_complete=True):
    """Return a single AMD lane report."""
    summary = summarize_events(events, payload_sizes, platform='rocprof', frame_count=frames)
    report = self_report_from_summary(summary, lane, 'rocprof', frames, profiler_complete)
    report['platform'] = 'rocprof'
    return report


def main():
    """Run the ROCprofiler parser/analyzer."""
    args = parse_args()
    try:
        if args.self_report:
            if not args.trace:
                raise ValueError('--self-report requires at least one --trace')
            if args.std_trace or args.managed_trace:
                raise ValueError('--self-report cannot be combined with paired trace arguments')
            result = self_report(
                load_events(args.trace), args.lane, set(args.payload_size),
                frames=args.frame_count,
            )
        else:
            if not args.std_trace or not args.managed_trace:
                raise ValueError('paired comparison requires --std-trace and --managed-trace')
            evidence = load_boundary_evidence(args.boundary_evidence)
            result = compare(
                load_events(args.std_trace),
                load_events(args.managed_trace),
                set(args.payload_size),
                std_frames=args.std_frames,
                managed_frames=args.managed_frames,
                adapter_directions=args.expected_adapter_direction or ('H2D', 'D2H'),
                boundary_evidence=evidence,
                kernel_payload_risk_names=args.kernel_payload_risk,
                binding_reports=load_binding_reports(
                    args.std_binding_report, args.managed_binding_report),
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if args.self_report:
        print(f"{result['lane']} ROCprofiler report written")
        return 0
    print(f"Status: {result['status']}")
    for direction, values in result['managed']['memory_totals'].items():
        print(
            f"Managed {direction}: count={values['count']} bytes={values['bytes']}")
    for item in result['adapter_copy_evidence']:
        print(
            'Expected Managed adapter copy: '
            f"{item['direction']} {item['bytes']} bytes x{item['count']}")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
