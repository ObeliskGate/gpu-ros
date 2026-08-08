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
Platform-neutral event normalization and transport-copy audit helpers.

Nsight Systems and ROCprofiler use different field names and different names
for the same host/device directions.  The audit tools deliberately normalize
only the evidence needed for the transport claim.  Kernel dispatches remain
separate from explicit memory-copy records: a kernel which happens to contain
``copy`` in its name is not, by itself, a copy failure.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

CPU_AGENT_WORDS = (
    'host', 'pageable', 'pinned', 'system', 'cpu', 'hsa_cpu', 'fine-grain',
)
DEVICE_AGENT_WORDS = (
    'device', 'gpu', 'cuda', 'hip', 'rocm', 'migraphx', 'hsa_gpu', 'agent',
)


def event_field(event: Mapping[str, Any], *names: str, default: Any = '') -> Any:
    """Read a case-insensitive field, accepting a list of aliases."""
    for name in names:
        if name in event:
            return event[name]
    lowered = {str(key).lower(): value for key, value in event.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def _number(value: Any) -> Optional[float]:
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(',', '').strip()
    if not text:
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def byte_count(event: Mapping[str, Any]) -> Optional[int]:
    """Return a record byte count, converting common unit-qualified columns."""
    value = event_field(
        event, 'bytes', 'byte_count', 'size', 'copy_size', 'transferred_bytes',
        'bytes_transferred', 'transfer_size', default=None)
    factor = 1.0
    if value is None or value == '':
        unit_factors = {
            'B': 1.0,
            'KB': 1000.0,
            'KIB': 1024.0,
            'MB': 1000.0 ** 2,
            'MIB': 1024.0 ** 2,
            'GB': 1000.0 ** 3,
            'GIB': 1024.0 ** 3,
        }
        for key, candidate in event.items():
            match = re.fullmatch(r'(?i)bytes\s*\(([^)]+)\)', str(key))
            if match and match.group(1).upper() in unit_factors:
                value = candidate
                factor = unit_factors[match.group(1).upper()]
                break
    elif isinstance(value, str):
        unit_factors = {
            'B': 1.0,
            'KB': 1000.0,
            'KIB': 1024.0,
            'MB': 1000.0 ** 2,
            'MIB': 1024.0 ** 2,
            'GB': 1000.0 ** 3,
            'GIB': 1024.0 ** 3,
        }
        unit_match = re.search(r'(?i)\b(KIB|KB|MIB|MB|GIB|GB|B)\b', value)
        if unit_match:
            factor = unit_factors[unit_match.group(1).upper()]
    number = _number(value)
    if number is None:
        return None
    return round(number * factor)


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _agent_text(value: Any) -> str:
    """Normalize ROCprofiler agent-id objects without losing the handle."""
    if isinstance(value, Mapping):
        for key in ('name', 'agent', 'handle', 'id'):
            if key in value:
                return _text(value[key])
    return _text(value)


def _agent_class(agent: str) -> str:
    text = agent.lower().replace('_', ' ').replace('-', ' ')
    if any(word in text for word in CPU_AGENT_WORDS):
        return 'host'
    if any(word in text for word in DEVICE_AGENT_WORDS):
        return 'device'
    return 'unknown'


def normalize_direction(
    source: Any,
    destination: Any,
    operation: Any = '',
    explicit_direction: Any = '',
) -> str:
    """Normalize H2D/D2H/D2D/H2H direction from agents or operation text."""
    source_text = _text(source).strip()
    destination_text = _text(destination).strip()
    explicit = _text(explicit_direction).strip().lower()
    operation_text = _text(operation).strip().lower()
    direction_text = f'{explicit} {operation_text}'

    patterns = (
        (r'(?:h|host|pageable|pinned)\s*(?:to|2|->)\s*(?:d|device|gpu)', 'H2D'),
        (r'(?:d|device|gpu)\s*(?:to|2|->)\s*(?:h|host|pageable|pinned)', 'D2H'),
        (r'(?:d|device|gpu)\s*(?:to|2|->)\s*(?:d|device|gpu)', 'D2D'),
        (r'(?:h|host|pageable|pinned)\s*(?:to|2|->)\s*(?:h|host|pageable|pinned)', 'H2H'),
    )
    compact = re.sub(r'[^a-z0-9>\-]+', '', direction_text)
    for pattern, direction in patterns:
        if re.search(pattern, direction_text) or re.search(pattern, compact):
            return direction

    source_class = _agent_class(source_text)
    destination_class = _agent_class(destination_text)
    if source_class == 'host' and destination_class == 'device':
        return 'H2D'
    if source_class == 'device' and destination_class == 'host':
        return 'D2H'
    if source_class == 'device' and destination_class == 'device':
        return 'D2D'
    if source_class == 'host' and destination_class == 'host':
        return 'H2H'
    return 'unknown'


def _is_copy_record(event: Mapping[str, Any], size: Optional[int]) -> bool:
    kind = _text(event_field(
        event, 'kind', 'category', 'record_type', 'record', 'domain', 'type', default=''))
    direction = _text(event_field(
        event, 'direction', 'copy_direction', 'transfer_direction', default=''))
    source = event_field(
        event, 'SrcMemKd', 'src_agent', 'source_agent', 'source', 'src', 'from',
        'src_agent_id', 'source_agent_id', default='')
    destination = event_field(
        event, 'DstMemKd', 'dst_agent', 'destination_agent', 'destination', 'dst', 'to',
        'dst_agent_id', 'destination_agent_id', default='')
    copy_words = r'copy|memcpy|memmove|memory[_ ]?operation|transfer|blit|h2d|d2h|d2d'
    kind_is_copy = bool(re.search(copy_words, kind.lower()))
    kind_is_kernel = bool(re.search(r'kernel|dispatch', kind.lower()))
    # A kernel is often named ``copy_*``.  Without a byte count, direction, or
    # agent fields that name alone is deliberately not treated as a memory-copy
    # record.  A byte count on a kernel row is also not sufficient; explicit
    # copy-row metadata must identify the record as a copy.
    if kind_is_copy and not kind_is_kernel:
        return True
    if direction or source or destination:
        return True
    if size is not None and kind_is_copy:
        return True
    # ROCprofiler memory-copy rows commonly carry only byte and agent fields;
    # the source/destination check above handles those rows.  Do not infer a
    # copy from the operation name alone because provider kernels may be named
    # ``copy_*`` or ``memcpy_*``.
    return False


def normalize_event(event: Mapping[str, Any], platform: str = 'generic') -> Dict[str, Any]:
    """Normalize one Nsight or ROCprofiler event."""
    del platform  # Field aliases are intentionally shared across platforms.
    operation = _text(event_field(
        event, 'Name', 'name', 'operation', 'op', 'kernel_name', 'kernel', 'event', default=''))
    size = byte_count(event)
    source = _agent_text(event_field(
        event, 'SrcMemKd', 'src_agent', 'source_agent', 'source', 'src', 'from',
        'src_agent_id', 'source_agent_id', default=''))
    destination = _agent_text(event_field(
        event, 'DstMemKd', 'dst_agent', 'destination_agent', 'destination', 'dst', 'to',
        'dst_agent_id', 'destination_agent_id', default=''))
    explicit_direction = event_field(
        event, 'direction', 'copy_direction', 'transfer_direction', default='')
    if _is_copy_record(event, size):
        event_type = 'memory_copy'
        direction = normalize_direction(source, destination, operation, explicit_direction)
    else:
        kind = _text(event_field(
            event, 'kind', 'category', 'record_type', 'domain', 'type', default='')).lower()
        kernel_name = _text(event_field(
            event, 'kernel_name', 'kernel', 'dispatch_name', 'name', 'Name', default=operation))
        if ('kernel' in kind or 'dispatch' in kind or
                event_field(event, 'kernel_name', 'dispatch_name', default=None) is not None):
            event_type = 'kernel'
            operation = kernel_name
        elif size is None and operation:
            # Nsight cuda_gpu_trace rows without a Bytes value are kernel rows.
            event_type = 'kernel'
            operation = kernel_name
        else:
            event_type = 'unknown'
    frame_id = event_field(
        event, 'frame', 'frame_id', 'frame_index', 'iteration', 'iteration_id', default=None)
    process_id = event_field(event, 'pid', 'process_id', 'process', default=None)
    payload_hint = bool(event_field(
        event, 'payload', 'payload_copy', 'tensor_payload', 'payload_candidate', default=False))
    return {
        'event_type': event_type,
        'operation': operation or '<unnamed>',
        'direction': direction if event_type == 'memory_copy' else '',
        'source_agent': source,
        'destination_agent': destination,
        'bytes': size,
        'frame_id': frame_id,
        'process_id': process_id,
        'payload_hint': payload_hint,
        'raw_fields': sorted(str(key) for key in event),
    }


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(keys.intersection({
        'name', 'operation', 'op', 'kernel_name', 'kernel', 'bytes', 'byte_count',
        'size', 'copy_size', 'srcmemkd', 'src_agent', 'source_agent', 'kind',
        'category', 'record_type', 'direction', 'timestamp', 'start', 'start_ns',
    }))


def iter_trace_records(document: Any) -> Iterator[Mapping[str, Any]]:
    """Yield record-like dictionaries from arrays, wrapper objects, or PID JSON."""
    if isinstance(document, list):
        for value in document:
            yield from iter_trace_records(value)
        return
    if not isinstance(document, dict):
        return
    container_keys = {
        'records', 'events', 'trace', 'data', 'processes', 'pid_data', 'results',
    }
    nested_containers = [
        value for key, value in document.items()
        if str(key).lower() in container_keys and isinstance(value, (dict, list))
    ]
    if nested_containers:
        for value in nested_containers:
            yield from iter_trace_records(value)
        return
    if _looks_like_record(document):
        yield document
        # A record can contain nested metadata, but nested dicts are not separate
        # trace events.  This avoids treating ``args`` as another kernel.
        return
    for value in document.values():
        if isinstance(value, (dict, list)):
            yield from iter_trace_records(value)


def load_trace_records(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    """Load and flatten one or more JSON trace files."""
    records: List[Mapping[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding='utf-8')
        if not text.strip():
            raise ValueError(f'{path}: report is empty')
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            first_line = text.splitlines()[0][:120]
            raise ValueError(
                f'{path}: report is not pure JSON; first line: {first_line!r}') from exc
        records.extend(iter_trace_records(document))
    return records


def signature(event: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return a stable, JSON-friendly-independent copy signature."""
    return (
        event.get('operation', '<unnamed>'),
        event.get('bytes'),
        event.get('direction', 'unknown'),
        event.get('source_agent', ''),
        event.get('destination_agent', ''),
    )


def _signature_json(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def serialize_counter(counter: Counter) -> List[Dict[str, Any]]:
    """Serialize a Counter with tuple keys in deterministic order."""
    return [
        {'key': _signature_json(key), 'count': counter[key]}
        for key in sorted(counter, key=str)
    ]


def counter_deltas(reference: Counter, candidate: Counter) -> List[Dict[str, Any]]:
    """Return candidate-minus-reference counter deltas."""
    return [
        {
            'key': _signature_json(key),
            'delta': candidate[key] - reference[key],
        }
        for key in sorted(set(reference) | set(candidate), key=str)
        if candidate[key] != reference[key]
    ]


def summarize_events(
    raw_events: Iterable[Mapping[str, Any]],
    payload_sizes: Iterable[int],
    platform: str = 'generic',
    frame_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize and summarize kernels, explicit copies, and frame rates."""
    normalized = [normalize_event(event, platform) for event in raw_events]
    payload_set: Set[int] = {int(size) for size in payload_sizes}
    kernels = Counter(
        event['operation'] for event in normalized if event['event_type'] == 'kernel')
    copies = Counter(
        signature(event)
        for event in normalized if event['event_type'] == 'memory_copy')
    memory_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {'count': 0, 'bytes': 0})
    payload_counts = Counter()
    frame_copy_counts: Dict[Any, Counter] = defaultdict(Counter)
    unknown_count = 0
    unknown_direction_copy_count = 0
    incomplete_memory_copy_records: List[Dict[str, Any]] = []
    for event in normalized:
        if event['event_type'] == 'unknown':
            unknown_count += 1
            continue
        if event['event_type'] != 'memory_copy':
            continue
        direction = event['direction'] or 'unknown'
        size = event['bytes']
        if direction == 'unknown':
            unknown_direction_copy_count += 1
        if size is None:
            incomplete_memory_copy_records.append({
                **_copy_record_for_json(event),
                'reason': 'memory_copy record has no byte count',
            })
        memory_totals[direction]['count'] += 1
        memory_totals[direction]['bytes'] += int(size or 0)
        sig = signature(event)
        if size in payload_set:
            payload_counts[(size, direction)] += 1
        if event['frame_id'] is not None:
            frame_copy_counts[event['frame_id']][sig] += 1

    denominator = frame_count if frame_count is not None else 0
    if denominator is None or denominator <= 0:
        denominator = len(frame_copy_counts) if frame_copy_counts else 0
    normalized_per_frame = {
        direction: {
            'count': data['count'] / denominator if denominator else None,
            'bytes': data['bytes'] / denominator if denominator else None,
        }
        for direction, data in sorted(memory_totals.items())
    }
    payload_per_frame = {
        f'{size}:{direction}': count / denominator if denominator else None
        for (size, direction), count in sorted(payload_counts.items(), key=str)
    }
    return {
        'event_count': len(normalized),
        'kernel_event_count': sum(kernels.values()),
        'kernels': kernels,
        'memcopies': copies,
        'memory_totals': dict(memory_totals),
        'payload_copy_counts': payload_counts,
        'normalized_per_frame': normalized_per_frame,
        'payload_normalized_per_frame': payload_per_frame,
        'frame_ids_observed': len(frame_copy_counts),
        'unknown_event_count': unknown_count,
        'unknown_direction_copy_count': unknown_direction_copy_count,
        'incomplete_memory_copy_records': incomplete_memory_copy_records,
        'normalized_events': normalized,
    }


def _is_adapter_direction(direction: str, adapter_directions: Set[str]) -> bool:
    return direction in adapter_directions


def _copy_record_for_json(event: Mapping[str, Any], count: int = 1) -> Dict[str, Any]:
    record = {
        'operation': event.get('operation', '<unnamed>'),
        'direction': event.get('direction', 'unknown'),
        'source_agent': event.get('source_agent', ''),
        'destination_agent': event.get('destination_agent', ''),
        'bytes': event.get('bytes'),
        'count': count,
    }
    if event.get('raw_fields'):
        record['raw_fields'] = event['raw_fields']
    return record


def _evidence_flags(boundary_evidence: Optional[Mapping[str, Any]]) -> Tuple[List[Any], List[Any]]:
    if not boundary_evidence:
        return [], []
    confirmed = boundary_evidence.get('confirmed_payload_copies', [])
    unresolved = boundary_evidence.get('unresolved_payload_copy_risk', [])
    if boundary_evidence.get('managed_boundary_payload_copy_confirmed'):
        confirmed = list(confirmed) + [
            {'reason': 'binding/lifetime evidence explicitly confirmed a boundary payload copy'}
        ]
    if isinstance(confirmed, bool):
        confirmed = ([{'reason': 'explicit boundary payload-copy confirmation'}]
                     if confirmed else [])
    if isinstance(unresolved, bool):
        unresolved = [{'reason': 'explicit unresolved payload-copy risk'}] if unresolved else []
    return list(confirmed), list(unresolved)


def looks_like_payload_kernel(name: str) -> bool:
    """Conservatively identify kernel names that need manual payload review."""
    return bool(re.search(r'copy|memcpy|memmove|blit|transfer', name.lower()))


def build_pair_report(
    reference_events: Iterable[Mapping[str, Any]],
    managed_events: Iterable[Mapping[str, Any]],
    payload_sizes: Iterable[int],
    reference_frames: Optional[int] = None,
    managed_frames: Optional[int] = None,
    max_payload_copy_rate_delta: float = 0.05,
    platform: str = 'generic',
    reference_lane: str = 'reference',
    managed_adapter_directions: Iterable[str] = (),
    require_adapter_directions: bool = False,
    boundary_evidence: Optional[Mapping[str, Any]] = None,
    profiler_complete: bool = True,
    kernel_payload_risk_names: Optional[Iterable[str]] = None,
    binding_reports: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the shared PASS/FAIL/INCONCLUSIVE transport audit report."""
    if (reference_frames is None) != (managed_frames is None):
        raise ValueError('both frame counts must be provided together')
    if reference_frames is not None and (reference_frames <= 0 or managed_frames <= 0):
        raise ValueError('frame counts must be positive')
    if max_payload_copy_rate_delta < 0:
        raise ValueError('max payload copy rate delta must be non-negative')

    payload_set = {int(size) for size in payload_sizes}
    binding_reports_provided = binding_reports is not None
    binding_reports = binding_reports or {}
    binding_report_errors: List[str] = []
    for lane in ('reference', 'managed') if binding_reports_provided else ():
        report = binding_reports.get(lane)
        if not isinstance(report, Mapping):
            binding_report_errors.append(f'{lane}: binding report missing')
            continue
        if report.get('first_frame') is not True:
            binding_report_errors.append(f'{lane}: first_frame marker missing')
        for side in ('inputs', 'outputs'):
            records = report.get(side)
            if not isinstance(records, list) or not records:
                binding_report_errors.append(f'{lane}: {side} records missing')
                continue
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    binding_report_errors.append(f'{lane}: {side}[{index}] is not an object')
                    continue
                for key in ('name', 'bytes', 'storage', 'ort_pointer', 'pointer_identity',
                            'lifetime_path'):
                    if key not in record:
                        binding_report_errors.append(f'{lane}: {side}[{index}] missing {key}')
                if not record.get('lifetime_path'):
                    binding_report_errors.append(f'{lane}: {side}[{index}] lifetime path empty')
    pointer_lifetime_complete = not binding_report_errors
    reference = summarize_events(reference_events, payload_set, platform, reference_frames)
    managed = summarize_events(managed_events, payload_set, platform, managed_frames)

    incomplete_memory_copy_evidence = []
    for lane, summary in (
        (reference_lane, reference),
        ('managed', managed),
    ):
        for record in summary['incomplete_memory_copy_records']:
            incomplete_memory_copy_evidence.append({
                'lane': lane,
                **record,
            })

    def has_explainable_events(summary):
        return summary['kernel_event_count'] > 0 or bool(summary['memcopies'])

    adapter_directions = {str(direction).upper() for direction in managed_adapter_directions}
    missing_adapter_directions = sorted(
        adapter_directions - set(managed['memory_totals'])
    ) if require_adapter_directions else []
    for direction in missing_adapter_directions:
        incomplete_memory_copy_evidence.append({
            'lane': 'managed',
            'operation': '<missing expected adapter direction>',
            'direction': direction,
            'source_agent': '',
            'destination_agent': '',
            'bytes': None,
            'count': 0,
            'reason': 'expected Managed adapter memory-copy direction was not observed',
        })
    trace_data_complete = (
        profiler_complete and has_explainable_events(reference) and
        has_explainable_events(managed) and
        not incomplete_memory_copy_evidence)
    memory_copy_failures: List[Dict[str, Any]] = []

    managed_extra = [
        (sig, count)
        for sig, count in managed['memcopies'].items()
        if sig not in reference['memcopies']
    ]
    for sig, count in managed_extra:
        direction = str(sig[2]).upper()
        if sig[1] in payload_set and not _is_adapter_direction(direction, adapter_directions):
            memory_copy_failures.append({
                'reason': 'managed-only tensor-sized memory_copy record',
                'signature': _signature_json(sig),
                'count': count,
            })

    payload_copy_rates: List[Dict[str, Any]] = []
    payload_rate_pass = True
    if reference_frames is not None:
        payload_keys = sorted(
            set(reference['payload_copy_counts']) | set(managed['payload_copy_counts']),
            key=str)
        for key in payload_keys:
            direction = str(key[1]).upper()
            reference_rate = reference['payload_copy_counts'][key] / reference_frames
            managed_rate = managed['payload_copy_counts'][key] / managed_frames
            rate_delta = managed_rate - reference_rate
            within_limit = (
                rate_delta <= max_payload_copy_rate_delta or
                _is_adapter_direction(direction, adapter_directions)
            )
            payload_rate_pass = payload_rate_pass and within_limit
            payload_copy_rates.append({
                'key': list(key),
                'config_c_per_frame': reference_rate,
                'managed_per_frame': managed_rate,
                'managed_minus_config_c': rate_delta,
                'within_limit': within_limit,
                'classification': (
                    'expected_managed_adapter'
                    if _is_adapter_direction(direction, adapter_directions)
                    else 'inference_boundary_candidate'
                ),
            })
            if not within_limit and key[0] in payload_set:
                memory_copy_failures.append({
                    'reason': 'managed-only tensor-sized payload copy rate exceeded limit',
                    'signature': [key[0], key[1]],
                    'managed_minus_reference_per_frame': rate_delta,
                })

    memory_directions = sorted(
        set(reference['memory_totals']) | set(managed['memory_totals']))
    memory_total_deltas = []
    for direction in memory_directions:
        reference_data = reference['memory_totals'].get(direction, {'count': 0, 'bytes': 0})
        managed_data = managed['memory_totals'].get(direction, {'count': 0, 'bytes': 0})
        memory_total_deltas.append({
            'direction': direction,
            'count_delta': managed_data['count'] - reference_data['count'],
            'byte_delta': managed_data['bytes'] - reference_data['bytes'],
            'classification': (
                'expected_managed_adapter'
                if _is_adapter_direction(direction, adapter_directions)
                else 'diagnostic'
            ),
        })

    managed_kernel_names = set(managed['kernels'])
    reference_kernel_names = set(reference['kernels'])
    kernel_only_differences = [
        {
            'name': name,
            'managed_count': managed['kernels'][name],
            'reference_count': reference['kernels'][name],
            'classification': 'diagnostic_kernel_set_difference',
        }
        for name in sorted(managed_kernel_names - reference_kernel_names)
    ]
    explicit_risk_names = set(kernel_payload_risk_names or ())
    unresolved_payload_copy_risk = [
        {
            'name': difference['name'],
            'reason': (
                'Managed-only kernel could carry tensor payload; kernel trace '
                'cannot prove otherwise'
            ),
        }
        for difference in kernel_only_differences
        if (
            difference['name'] in explicit_risk_names or
            looks_like_payload_kernel(difference['name'])
        )
    ]
    _, evidence_unresolved = _evidence_flags(boundary_evidence)
    unresolved_payload_copy_risk.extend(evidence_unresolved)
    confirmed_boundary, _ = _evidence_flags(boundary_evidence)
    boundary_payload_copy_evidence = [
        {
            'classification': 'confirmed_boundary_payload_copy',
            'evidence': item,
        }
        for item in confirmed_boundary
    ]

    adapter_copy_evidence = []
    for sig, managed_count in managed['memcopies'].items():
        if str(sig[2]).upper() not in adapter_directions:
            continue
        evidence = _copy_record_for_json(
            {'operation': sig[0], 'bytes': sig[1], 'direction': sig[2],
             'source_agent': sig[3], 'destination_agent': sig[4]}, managed_count)
        evidence['reference_count'] = reference['memcopies'].get(sig, 0)
        evidence['managed_only'] = sig not in reference['memcopies']
        adapter_copy_evidence.append(evidence)

    # Explicit boundary evidence outranks all heuristic classification.
    for item in confirmed_boundary:
        memory_copy_failures.append({
            'reason': (
                'pointer/lifetime and trace evidence confirmed a Managed boundary '
                'payload copy'
            ),
            'evidence': item,
        })

    boundary_payload_copy_evidence.extend({
        'classification': 'explicit_memory_copy_record',
        'evidence': failure,
    } for failure in memory_copy_failures if 'tensor-sized' in failure.get('reason', ''))

    if memory_copy_failures:
        status = 'FAIL'
    elif (
        not trace_data_complete or pointer_lifetime_complete is False or
        unresolved_payload_copy_risk
    ):
        status = 'INCONCLUSIVE'
    else:
        status = 'PASS'

    managed_extra_memcopies = [
        {'signature': _signature_json(sig), 'count': count}
        for sig, count in managed_extra
    ]
    signature_deltas = counter_deltas(reference['memcopies'], managed['memcopies'])
    kernel_name_sets_match = reference_kernel_names == managed_kernel_names
    memory_copy_evidence_complete = trace_data_complete
    result = {
        'schema_version': 2,
        'status': status,
        'final_status': status,
        'pass': status == 'PASS',
        'reference_lane': reference_lane,
        'bridge_zero_copy_pass': not memory_copy_failures,
        # Kept for historical consumers.  This is diagnostic, not a closure
        # criterion; a provider may legitimately fuse or split kernels.
        'gpu_execution_control_pass': kernel_name_sets_match,
        'criteria': {
            'kernel_name_sets_match': kernel_name_sets_match,
            'managed_has_no_extra_memcpy_signature': not managed_extra_memcopies,
            'managed_has_no_extra_payload_copy_rate': payload_rate_pass,
            'memory_copy_evidence_complete': memory_copy_evidence_complete,
            'memory_copy_records_explainable': not incomplete_memory_copy_evidence,
            'expected_adapter_directions_observed': not missing_adapter_directions,
            'profiler_data_complete': trace_data_complete,
            'pointer_lifetime_evidence_complete': pointer_lifetime_complete,
            'kernel_differences_are_diagnostic': True,
            'cpu_fallback_is_diagnostic_only': True,
        },
        'self': {
            'reference': self_report_from_summary(
                reference, reference_lane, platform, reference_frames, profiler_complete),
            'managed': self_report_from_summary(
                managed, 'managed', platform, managed_frames, profiler_complete),
        },
        # These two top-level sections make paired reports useful without
        # requiring consumers to know the historical field names.
        reference_lane: self_report_from_summary(
            reference, reference_lane, platform, reference_frames, profiler_complete),
        'managed': self_report_from_summary(
            managed, 'managed', platform, managed_frames, profiler_complete),
        'kernel_only_differences': kernel_only_differences,
        'unresolved_payload_copy_risk': unresolved_payload_copy_risk,
        'unresolved_memory_copy_evidence': incomplete_memory_copy_evidence,
        'pointer_lifetime_evidence': {
            'complete': pointer_lifetime_complete,
            'errors': binding_report_errors,
            'reports': dict(binding_reports),
        },
        'boundary_payload_copy_evidence': boundary_payload_copy_evidence,
        'memory_copy_failures': memory_copy_failures,
        'adapter_copy_evidence': adapter_copy_evidence,
        'memory_copy': {
            'reference': serialize_counter(reference['memcopies']),
            'managed': serialize_counter(managed['memcopies']),
            'signature_count_deltas': signature_deltas,
            'managed_extra_signatures': managed_extra_memcopies,
            'memory_total_deltas': memory_total_deltas,
            'unresolved_records': incomplete_memory_copy_evidence,
            'normalized_per_frame': {
                reference_lane: reference['normalized_per_frame'],
                'managed': managed['normalized_per_frame'],
            },
        },
        # Historical spelling retained for existing reports and downstream
        # scripts.
        'memcpy': {
            'reference': serialize_counter(reference['memcopies']),
            'config_c': serialize_counter(reference['memcopies']),
            'managed': serialize_counter(managed['memcopies']),
            'signature_count_deltas': signature_deltas,
            'managed_extra_signatures': managed_extra_memcopies,
            'memory_total_deltas': memory_total_deltas,
        },
        'kernel_names': {
            'config_c_unique_count': len(reference_kernel_names),
            'managed_unique_count': len(managed_kernel_names),
            'missing_from_managed': sorted(reference_kernel_names - managed_kernel_names),
            'extra_in_managed': sorted(managed_kernel_names - reference_kernel_names),
            'event_count_deltas': counter_deltas(reference['kernels'], managed['kernels']),
        },
        'payload_copy_counts': {
            'sizes_bytes': sorted(payload_set),
            'config_c_frames': reference_frames,
            'managed_frames': managed_frames,
            'max_managed_minus_config_c_per_frame': max_payload_copy_rate_delta,
            'config_c': serialize_counter(reference['payload_copy_counts']),
            'managed': serialize_counter(managed['payload_copy_counts']),
            'deltas': counter_deltas(
                reference['payload_copy_counts'], managed['payload_copy_counts']),
            'normalized_rates': payload_copy_rates,
        },
    }
    return result


def self_report_from_summary(
    summary: Mapping[str, Any],
    lane: str,
    platform: str,
    frames: Optional[int],
    profiler_complete: bool,
) -> Dict[str, Any]:
    """Serialize one lane without exposing Python Counter objects."""
    effective_profiler_complete = profiler_complete and (
        summary['kernel_event_count'] > 0 or bool(summary['memcopies']))
    return {
        'schema_version': 2,
        'report_type': 'self',
        'lane': lane,
        'platform': platform,
        'profiler_complete': effective_profiler_complete,
        'event_count': summary['event_count'],
        'kernel_event_count': summary['kernel_event_count'],
        'kernels': serialize_counter(summary['kernels']),
        'memory_copy': serialize_counter(summary['memcopies']),
        'memory_totals': summary['memory_totals'],
        'payload_copy_counts': serialize_counter(summary['payload_copy_counts']),
        'frame_count': frames,
        'frame_ids_observed': summary['frame_ids_observed'],
        'normalized_per_frame': summary['normalized_per_frame'],
        'payload_normalized_per_frame': summary['payload_normalized_per_frame'],
        'unknown_event_count': summary['unknown_event_count'],
        'unknown_direction_copy_count': summary['unknown_direction_copy_count'],
        'incomplete_memory_copy_records': summary['incomplete_memory_copy_records'],
    }
