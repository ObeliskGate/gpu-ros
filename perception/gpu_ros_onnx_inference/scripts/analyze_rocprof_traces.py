#!/usr/bin/env python3
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

"""Analyze ROCprofiler-v3 JSON evidence for the AMD Managed boundary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from copy_audit_common import (
        build_pair_report,
        self_report_from_summary,
        summarize_events,
    )
    from rocprofv3_json_loader import (
        KERNEL_TRACE_DOMAIN,
        MEMORY_COPY_DOMAIN,
        RocprofCapture,
        RocprofJsonError,
        RocprofManifestError,
        capture_completeness,
        cross_check_csv,
        load_manifest,
        load_rocprof_json,
    )
except ModuleNotFoundError:  # Direct source-tree imports used by pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from copy_audit_common import (  # type: ignore[no-redef]
        build_pair_report,
        self_report_from_summary,
        summarize_events,
    )
    from rocprofv3_json_loader import (  # type: ignore[no-redef]
        KERNEL_TRACE_DOMAIN,
        MEMORY_COPY_DOMAIN,
        RocprofCapture,
        RocprofJsonError,
        RocprofManifestError,
        capture_completeness,
        cross_check_csv,
        load_manifest,
        load_rocprof_json,
    )


def parse_args() -> argparse.Namespace:
    """Parse paired or self-report arguments."""
    parser = argparse.ArgumentParser(
        description='Parse official ROCprofiler-v3 JSON activity sections.')
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
        help='Direction expected from Managed adapter staging; not proof by itself')
    parser.add_argument(
        '--require-adapter-directions', action='store_true',
        help='Require every expected adapter direction (staged-control lane only)')
    parser.add_argument('--boundary-evidence', type=Path)
    parser.add_argument('--std-binding-report', type=Path)
    parser.add_argument('--managed-binding-report', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--std-manifest', type=Path)
    parser.add_argument('--managed-manifest', type=Path)
    parser.add_argument('--csv', action='append', type=Path)
    parser.add_argument('--std-csv', action='append', type=Path)
    parser.add_argument('--managed-csv', action='append', type=Path)
    parser.add_argument(
        '--kernel-payload-risk', action='append', default=[],
        help='Managed-only kernel name that cannot be ruled out as payload movement')
    parser.add_argument('--output-json', required=True, type=Path)
    return parser.parse_args()


def load_capture(paths: Sequence[Path]) -> RocprofCapture:
    """Load one or more official ROCprofiler JSON files."""
    return load_rocprof_json([Path(path) for path in paths])


def load_events(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    """Compatibility helper returning only canonical events."""
    return load_capture(paths).events


def summarize(
        events: Iterable[Mapping[str, Any]],
        payload_sizes: Iterable[int],
        frames: Optional[int] = None) -> Dict[str, Any]:
    """Summarize canonical events, primarily for offline tests/notebooks."""
    # ROCprof-specific enum and metadata handling has already happened in the
    # dedicated loader.  The common summarizer sees ordinary canonical events.
    return summarize_events(events, payload_sizes, platform='generic', frame_count=frames)


def load_boundary_evidence(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if path is None:
        return None
    try:
        with path.open(encoding='utf-8') as evidence_file:
            evidence = json.load(evidence_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RocprofJsonError(f'{path}: invalid boundary evidence: {exc}') from exc
    if not isinstance(evidence, Mapping):
        raise RocprofJsonError(f'{path}: boundary evidence must be a JSON object')
    return evidence


def load_binding_reports(
        std_path: Optional[Path],
        managed_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load first-frame pointer/lifetime reports for both AMD lanes."""
    if std_path is None and managed_path is None:
        return None
    reports: Dict[str, Any] = {}
    for lane, path in (('reference', std_path), ('managed', managed_path)):
        if path is None:
            reports[lane] = None
            continue
        try:
            with path.open(encoding='utf-8') as report_file:
                reports[lane] = json.load(report_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RocprofJsonError(f'{path}: invalid binding report: {exc}') from exc
    return reports


def _serialize_counter(counter: Counter) -> List[Dict[str, Any]]:
    return [
        {'key': list(key) if isinstance(key, tuple) else key, 'count': counter[key]}
        for key in sorted(counter, key=str)
    ]


def _kernel_summary(
        events: Iterable[Mapping[str, Any]],
        frames: Optional[int]) -> Dict[str, Any]:
    summary = summarize_events(events, set(), platform='generic', frame_count=frames)
    kernels = summary['kernels']
    return {
        'total_count': sum(kernels.values()),
        'unique_count': len(kernels),
        'counts': _serialize_counter(kernels),
        'per_frame': {
            name: (count / frames if frames else None)
            for name, count in sorted(kernels.items())
        },
        'frame_count': frames,
    }


def _kernel_evidence(
        std_events: Iterable[Mapping[str, Any]],
        managed_events: Iterable[Mapping[str, Any]],
        std_frames: Optional[int],
        managed_frames: Optional[int],
        boundary_evidence: Optional[Mapping[str, Any]],
        managed_only_risk_names: Iterable[str]) -> Dict[str, Any]:
    """Build AMD-only kernel evidence without turning names into copy facts."""
    std_events = list(std_events)
    managed_events = list(managed_events)
    std_summary = _kernel_summary(std_events, std_frames)
    managed_summary = _kernel_summary(managed_events, managed_frames)
    std_counts = Counter({item['key']: item['count'] for item in std_summary['counts']})
    managed_counts = Counter({item['key']: item['count'] for item in managed_summary['counts']})
    std_names = set(std_counts)
    managed_names = set(managed_counts)
    shared_names = std_names & managed_names
    count_deltas = []
    for name in sorted(std_names | managed_names):
        std_count = std_counts[name]
        managed_count = managed_counts[name]
        count_deltas.append({
            'name': name,
            'std_count': std_count,
            'managed_count': managed_count,
            'count_delta': managed_count - std_count,
            'std_per_frame': std_count / std_frames if std_frames else None,
            'managed_per_frame': managed_count / managed_frames if managed_frames else None,
            'rate_delta': (
                managed_count / managed_frames - std_count / std_frames
                if std_frames and managed_frames else None),
            'classification': 'shared_kernel_rate_diagnostic' if name in shared_names
            else 'kernel_set_diagnostic',
        })

    copy_buffer = [
        item for item in count_deltas
        if '__amd_rocclr_copyBuffer' in item['name']
    ]

    # A trace/correlation artifact may explicitly explain a kernel's relation
    # to the inference boundary.  Without that artifact, a shared
    # copyBuffer-rate difference is AMD-specific unresolved evidence, not a
    # generic copy failure.
    resolved_names = set()
    if isinstance(boundary_evidence, Mapping):
        for key in ('resolved_kernel_names', 'explained_kernel_names'):
            values = boundary_evidence.get(key, [])
            if isinstance(values, list):
                resolved_names.update(str(value) for value in values)
    unresolved: List[Dict[str, Any]] = []
    for item in copy_buffer:
        if item['count_delta'] and item['name'] not in resolved_names:
            unresolved.append({
                'name': item['name'],
                'reason': (
                    'shared AMD copyBuffer kernel rate differs and the available '
                    'trace/correlation does not establish its relation to the '
                    'inference boundary'
                ),
                'std_count': item['std_count'],
                'managed_count': item['managed_count'],
                'rate_delta': item['rate_delta'],
                'classification': 'unresolved_kernel_evidence',
            })
    for name in sorted(set(managed_only_risk_names)):
        if name in managed_names and name not in resolved_names:
            unresolved.append({
                'name': name,
                'reason': (
                    'explicitly supplied Managed-only kernel payload risk has no '
                    'boundary correlation'
                ),
                'std_count': std_counts[name],
                'managed_count': managed_counts[name],
                'classification': 'unresolved_kernel_evidence',
            })
    return {
        'std': std_summary,
        'managed': managed_summary,
        'kernel_set': {
            'shared_count': len(shared_names),
            'std_only': sorted(std_names - managed_names),
            'managed_only': sorted(managed_names - std_names),
        },
        'count_deltas': count_deltas,
        'copyBuffer': copy_buffer,
        'unresolved_kernel_evidence': unresolved,
    }


def _capture_details(
        capture: Optional[RocprofCapture],
        manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if capture is None:
        return {
            'memory_copy': {'domain': MEMORY_COPY_DOMAIN, 'status': 'not_checked'},
            'kernel': {'domain': KERNEL_TRACE_DOMAIN, 'status': 'not_checked'},
            'parser_diagnostic_count': 0,
            'parser_diagnostics': [],
            'memory_copy_complete': True,
            'kernel_complete': True,
        }
    return capture_completeness(capture, manifest)


def _capture_profiler_complete(details: Mapping[str, Any]) -> bool:
    return bool(details.get('memory_copy_complete')) and not details.get(
        'parser_diagnostics')


def _add_amd_classifications(result: Dict[str, Any]) -> None:
    """Annotate historical adapter evidence without changing common rules."""
    for item in result.get('adapter_copy_evidence', []):
        item['classification'] = 'staging_shaped_evidence'
        item['interpretation'] = (
            'Direction/bytes identify host-device-shaped staging, but do not '
            'prove that this record belongs to the Managed adapter.')
    for item in result.get('memory_copy', {}).get('memory_total_deltas', []):
        if item.get('classification') == 'expected_managed_adapter':
            item['classification'] = 'staging_shaped_evidence'


def format_unresolved_kernel(item: Mapping[str, Any]) -> str:
    """Format named kernel evidence and incomplete-domain records safely."""
    if item.get('name'):
        return f"name={item['name']} reason={item.get('reason', 'unspecified')}"
    return (
        f"lane={item.get('lane', 'unknown')} "
        f"classification={item.get('classification', 'unresolved_kernel_evidence')} "
        f"reason={item.get('reason', 'unspecified')}")


def compare(
        std_events: Iterable[Mapping[str, Any]],
        managed_events: Iterable[Mapping[str, Any]],
        payload_sizes: Iterable[int],
        std_frames: Optional[int] = None,
        managed_frames: Optional[int] = None,
        adapter_directions: Iterable[str] = ('H2D', 'D2H'),
        require_adapter_directions: bool = False,
        boundary_evidence: Optional[Mapping[str, Any]] = None,
        profiler_complete: bool = True,
        kernel_payload_risk_names: Optional[Iterable[str]] = None,
        binding_reports: Optional[Mapping[str, Mapping[str, Any]]] = None,
        std_capture: Optional[RocprofCapture] = None,
        managed_capture: Optional[RocprofCapture] = None,
        std_manifest: Optional[Mapping[str, Any]] = None,
        managed_manifest: Optional[Mapping[str, Any]] = None,
        std_csv: Optional[Sequence[Path]] = None,
        managed_csv: Optional[Sequence[Path]] = None,
        ) -> Dict[str, Any]:
    """Build an AMD report while keeping common NVIDIA criteria unchanged."""
    std_events = list(std_events)
    managed_events = list(managed_events)
    std_details = _capture_details(std_capture, std_manifest)
    managed_details = _capture_details(managed_capture, managed_manifest)
    complete = (
        profiler_complete and
        _capture_profiler_complete(std_details) and
        _capture_profiler_complete(managed_details))
    kernel_evidence = _kernel_evidence(
        std_events, managed_events, std_frames, managed_frames,
        boundary_evidence, kernel_payload_risk_names or ())
    result = build_pair_report(
        std_events,
        managed_events,
        payload_sizes,
        reference_frames=std_frames,
        managed_frames=managed_frames,
        # Events from the dedicated loader are already canonical.  Passing
        # generic here is intentional: ROCprof enum handling must not leak into
        # copy_audit_common or change NVIDIA/Nsight behavior.
        platform='generic',
        reference_lane='std',
        managed_adapter_directions=adapter_directions,
        require_adapter_directions=require_adapter_directions,
        boundary_evidence=boundary_evidence,
        profiler_complete=complete,
        kernel_payload_risk_names=kernel_payload_risk_names,
        binding_reports=binding_reports,
    )
    result['platform'] = 'rocprof'
    result['capture_completeness'] = {
        'std': std_details,
        'managed': managed_details,
    }
    result['kernel_evidence'] = kernel_evidence
    unresolved_kernel_evidence = list(kernel_evidence['unresolved_kernel_evidence'])
    for lane, details in (('std', std_details), ('managed', managed_details)):
        kernel_status = details['kernel']['status']
        if kernel_status not in {'complete', 'not_checked'}:
            unresolved_kernel_evidence.append({
                'lane': lane,
                'reason': (
                    f'ROCprofiler kernel domain is {kernel_status}; kernel trace '
                    'is auxiliary evidence and cannot explain the AMD boundary'),
                'classification': 'unresolved_kernel_evidence',
            })
    result['unresolved_kernel_evidence'] = unresolved_kernel_evidence
    result['staging_policy'] = {
        'managed_adapter_directions': sorted(set(adapter_directions)),
        'description': (
            'Managed-only H2D/D2H records are reported as staging-shaped '
            'evidence. Direction and bytes alone do not prove adapter ownership; '
            'explicit boundary correlation is required for a boundary FAIL.'),
    }
    _add_amd_classifications(result)

    csv_checks = {}
    if std_capture is not None and std_csv:
        csv_checks['std'] = cross_check_csv(std_capture, std_csv)
    if managed_capture is not None and managed_csv:
        csv_checks['managed'] = cross_check_csv(managed_capture, managed_csv)
    if csv_checks:
        result['csv_cross_check'] = csv_checks
        if any(item['status'] != 'PASS' for item in csv_checks.values()) and \
                result['status'] != 'FAIL':
            result['status'] = 'INCONCLUSIVE'
            result['final_status'] = 'INCONCLUSIVE'
            result['pass'] = False

    if result['status'] != 'FAIL' and result['unresolved_kernel_evidence']:
        result['status'] = 'INCONCLUSIVE'
        result['final_status'] = 'INCONCLUSIVE'
        result['pass'] = False
    result['criteria']['amd_kernel_evidence_resolved'] = not bool(
        result['unresolved_kernel_evidence'])
    result['criteria']['memory_copy_manifest_complete'] = (
        std_details['memory_copy']['status'] in {'complete', 'not_checked'} and
        managed_details['memory_copy']['status'] in {'complete', 'not_checked'})
    result['criteria']['kernel_manifest_is_auxiliary'] = True
    return result


def self_report(
        events: Iterable[Mapping[str, Any]],
        lane: str,
        payload_sizes: Iterable[int],
        frames: Optional[int] = None,
        profiler_complete: bool = True,
        capture: Optional[RocprofCapture] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        csv_paths: Optional[Sequence[Path]] = None,
        ) -> Dict[str, Any]:
    """Return one lane's canonical report."""
    events = list(events)
    details = _capture_details(capture, manifest)
    effective_complete = profiler_complete and _capture_profiler_complete(details)
    summary = summarize_events(events, payload_sizes, platform='generic', frame_count=frames)
    report = self_report_from_summary(summary, lane, 'rocprof', frames, effective_complete)
    report['platform'] = 'rocprof'
    report['capture_completeness'] = details
    report['kernel_evidence'] = _kernel_evidence(
        [], events, None, frames, None, ())['managed']
    if capture is not None:
        report['parser_diagnostics'] = capture.diagnostics
        if csv_paths:
            report['csv_cross_check'] = cross_check_csv(capture, csv_paths)
    return report


def _load_manifest_optional(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    return load_manifest(path) if path is not None else None


def main() -> int:
    """Run the ROCprofiler parser/analyzer."""
    args = parse_args()
    try:
        if args.self_report:
            if not args.trace:
                raise RocprofJsonError('--self-report requires at least one --trace')
            if args.std_trace or args.managed_trace:
                raise RocprofJsonError(
                    '--self-report cannot be combined with paired trace arguments')
            capture = load_capture(args.trace)
            result = self_report(
                capture.events, args.lane, set(args.payload_size),
                frames=args.frame_count,
                capture=capture,
                manifest=_load_manifest_optional(args.manifest),
                csv_paths=args.csv,
            )
        else:
            if not args.std_trace or not args.managed_trace:
                raise RocprofJsonError(
                    'paired comparison requires --std-trace and --managed-trace')
            std_capture = load_capture(args.std_trace)
            managed_capture = load_capture(args.managed_trace)
            result = compare(
                std_capture.events,
                managed_capture.events,
                set(args.payload_size),
                std_frames=args.std_frames,
                managed_frames=args.managed_frames,
                adapter_directions=args.expected_adapter_direction or ('H2D', 'D2H'),
                require_adapter_directions=args.require_adapter_directions,
                boundary_evidence=load_boundary_evidence(args.boundary_evidence),
                kernel_payload_risk_names=args.kernel_payload_risk,
                binding_reports=load_binding_reports(
                    args.std_binding_report, args.managed_binding_report),
                std_capture=std_capture,
                managed_capture=managed_capture,
                std_manifest=_load_manifest_optional(args.std_manifest),
                managed_manifest=_load_manifest_optional(args.managed_manifest),
                std_csv=args.std_csv,
                managed_csv=args.managed_csv,
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError,
            RocprofJsonError, RocprofManifestError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if args.self_report:
        print(f"{result['lane']} ROCprofiler report written")
        return 0
    print(f"Status: {result['status']}")
    for lane in (result['reference_lane'], 'managed'):
        for direction, values in result[lane]['memory_totals'].items():
            print(f"{lane} {direction}: count={values['count']} bytes={values['bytes']}")
    for domain_lane in ('std', 'managed'):
        details = result['capture_completeness'][domain_lane]
        print(f"{domain_lane} memory-copy domain: {details['memory_copy']['status']}")
    for item in result.get('unresolved_memory_copy_evidence', []):
        print(
            'Unresolved memory-copy record: '
            f"lane={item['lane']} operation={item['operation']} "
            f"direction={item['direction']} bytes={item['bytes']} "
            f"reason={item['reason']}")
    for item in result.get('unresolved_kernel_evidence', []):
        print('Unresolved AMD kernel evidence: ' + format_unresolved_kernel(item))
    for item in result.get('adapter_copy_evidence', []):
        print(
            'Staging-shaped Managed copy evidence: '
            f"{item['direction']} {item['bytes']} bytes x{item['count']}")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
