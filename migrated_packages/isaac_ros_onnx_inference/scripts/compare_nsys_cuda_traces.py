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

"""Compare C/M CUDA kernels and memory operations exported by Nsight Systems."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Config C and Managed nsys cuda_gpu_trace JSON files.')
    parser.add_argument('--config-c-trace', required=True, type=Path)
    parser.add_argument('--managed-trace', required=True, type=Path)
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
    return parser.parse_args()


def load_events(path):
    """Load one nsys JSON report."""
    trace_text = path.read_text(encoding='utf-8')
    if not trace_text.strip():
        raise ValueError(f'{path}: report is empty')
    try:
        events = json.loads(trace_text)
    except json.JSONDecodeError as exc:
        first_line = trace_text.splitlines()[0][:120]
        raise ValueError(
            f'{path}: report is not pure JSON; first line: {first_line!r}') from exc
    if not isinstance(events, list):
        raise ValueError(f'{path}: top-level JSON value must be an array')
    return [event for event in events if isinstance(event, dict)]


def event_field(event, name, default=''):
    """Read a field while tolerating formatter capitalization changes."""
    if name in event:
        return event[name]
    lower_name = name.lower()
    for key, value in event.items():
        if key.lower() == lower_name:
            return value
    return default


def byte_count(event):
    """Return a memory-operation byte count, or None for a kernel row."""
    value = event_field(event, 'Bytes', '')
    factor = 1
    if value in ('', None):
        unit_factors = {
            'B': 1,
            'KB': 1000,
            'KiB': 1024,
            'MB': 1000 ** 2,
            'MiB': 1024 ** 2,
            'GB': 1000 ** 3,
            'GiB': 1024 ** 3,
        }
        for key, candidate in event.items():
            if not key.startswith('Bytes (') or not key.endswith(')'):
                continue
            unit = key[len('Bytes ('):-1]
            if unit in unit_factors:
                value = candidate
                factor = unit_factors[unit]
                break
    if value in ('', None):
        return None
    if isinstance(value, str):
        value = value.replace(',', '').strip()
        if not value:
            return None
    return int(round(float(value) * factor))


def summarize(events, payload_sizes):
    """Summarize stable kernel names and memcpy signatures."""
    kernels = Counter()
    memcopies = Counter()
    memory_totals = defaultdict(lambda: {'count': 0, 'bytes': 0})
    payload_copy_counts = Counter()

    for event in events:
        name = str(event_field(event, 'Name', '<unnamed>'))
        size = byte_count(event)
        if size is None:
            kernels[name] += 1
            continue

        source = str(event_field(event, 'SrcMemKd', ''))
        destination = str(event_field(event, 'DstMemKd', ''))
        direction = f'{source}->{destination}'
        if 'memcpy' not in name.lower():
            continue

        memory_totals[direction]['count'] += 1
        memory_totals[direction]['bytes'] += size
        signature = (name, size, source, destination)
        memcopies[signature] += 1
        if size in payload_sizes:
            payload_copy_counts[(size, direction)] += 1

    return {
        'kernels': kernels,
        'memcopies': memcopies,
        'memory_totals': dict(memory_totals),
        'payload_copy_counts': payload_copy_counts,
    }


def counter_deltas(reference, candidate):
    """Return sorted non-zero candidate-minus-reference counter deltas."""
    return [
        {
            'key': list(key) if isinstance(key, tuple) else key,
            'delta': candidate[key] - reference[key],
        }
        for key in sorted(set(reference) | set(candidate), key=str)
        if candidate[key] != reference[key]
    ]


def serialize_counter(counter):
    """Convert a possibly tuple-keyed Counter into stable JSON records."""
    return [
        {'key': list(key) if isinstance(key, tuple) else key, 'count': counter[key]}
        for key in sorted(counter, key=str)
    ]


def compare(
        config_c_events,
        managed_events,
        payload_sizes,
        config_c_frames=None,
        managed_frames=None,
        max_payload_copy_rate_delta=0.05):
    """Build the machine-readable transport audit comparison."""
    if (config_c_frames is None) != (managed_frames is None):
        raise ValueError('both frame counts must be provided together')
    if config_c_frames is not None and (config_c_frames <= 0 or managed_frames <= 0):
        raise ValueError('frame counts must be positive')
    if max_payload_copy_rate_delta < 0:
        raise ValueError('max payload copy rate delta must be non-negative')

    config_c = summarize(config_c_events, payload_sizes)
    managed = summarize(managed_events, payload_sizes)
    c_kernel_names = set(config_c['kernels'])
    m_kernel_names = set(managed['kernels'])
    memcpy_deltas = counter_deltas(config_c['memcopies'], managed['memcopies'])
    managed_extra_memcopies = serialize_counter(Counter({
        signature: count
        for signature, count in managed['memcopies'].items()
        if signature not in config_c['memcopies']
    }))

    payload_copy_rates = []
    payload_copy_rate_pass = True
    if config_c_frames is not None:
        payload_keys = sorted(
            set(config_c['payload_copy_counts']) |
            set(managed['payload_copy_counts']),
            key=str)
        for key in payload_keys:
            config_c_rate = config_c['payload_copy_counts'][key] / config_c_frames
            managed_rate = managed['payload_copy_counts'][key] / managed_frames
            rate_delta = managed_rate - config_c_rate
            within_limit = rate_delta <= max_payload_copy_rate_delta
            payload_copy_rate_pass = payload_copy_rate_pass and within_limit
            payload_copy_rates.append({
                'key': list(key),
                'config_c_per_frame': config_c_rate,
                'managed_per_frame': managed_rate,
                'managed_minus_config_c': rate_delta,
                'within_limit': within_limit,
            })

    memory_directions = sorted(
        set(config_c['memory_totals']) | set(managed['memory_totals']))
    memory_total_deltas = []
    for direction in memory_directions:
        c_data = config_c['memory_totals'].get(direction, {'count': 0, 'bytes': 0})
        m_data = managed['memory_totals'].get(direction, {'count': 0, 'bytes': 0})
        memory_total_deltas.append({
            'direction': direction,
            'count_delta': m_data['count'] - c_data['count'],
            'byte_delta': m_data['bytes'] - c_data['bytes'],
        })

    bridge_zero_copy_pass = not managed_extra_memcopies and payload_copy_rate_pass
    gpu_execution_control_pass = c_kernel_names == m_kernel_names
    result = {
        'pass': bridge_zero_copy_pass and gpu_execution_control_pass,
        'bridge_zero_copy_pass': bridge_zero_copy_pass,
        'gpu_execution_control_pass': gpu_execution_control_pass,
        'criteria': {
            'kernel_name_sets_match': c_kernel_names == m_kernel_names,
            'managed_has_no_extra_memcpy_signature': not managed_extra_memcopies,
            'managed_has_no_extra_payload_copy_rate': payload_copy_rate_pass,
        },
        'kernel_names': {
            'config_c_unique_count': len(c_kernel_names),
            'managed_unique_count': len(m_kernel_names),
            'missing_from_managed': sorted(c_kernel_names - m_kernel_names),
            'extra_in_managed': sorted(m_kernel_names - c_kernel_names),
            'event_count_deltas': counter_deltas(
                config_c['kernels'], managed['kernels']),
        },
        'memcpy': {
            'config_c': serialize_counter(config_c['memcopies']),
            'managed': serialize_counter(managed['memcopies']),
            'signature_count_deltas': memcpy_deltas,
            'managed_extra_signatures': managed_extra_memcopies,
            'memory_total_deltas': memory_total_deltas,
        },
        'payload_copy_counts': {
            'sizes_bytes': sorted(payload_sizes),
            'config_c_frames': config_c_frames,
            'managed_frames': managed_frames,
            'max_managed_minus_config_c_per_frame': max_payload_copy_rate_delta,
            'config_c': serialize_counter(config_c['payload_copy_counts']),
            'managed': serialize_counter(managed['payload_copy_counts']),
            'deltas': counter_deltas(
                config_c['payload_copy_counts'], managed['payload_copy_counts']),
            'normalized_rates': payload_copy_rates,
        },
    }
    return result


def main():
    """Run the CUDA trace comparison."""
    args = parse_args()
    try:
        result = compare(
            load_events(args.config_c_trace),
            load_events(args.managed_trace),
            set(args.payload_size),
            args.config_c_frames,
            args.managed_frames,
            args.max_payload_copy_rate_delta,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f"Kernel name sets match: {result['criteria']['kernel_name_sets_match']}")
    print(
        'Managed has no extra memcpy signature: '
        f"{result['criteria']['managed_has_no_extra_memcpy_signature']}")
    print(
        'Managed has no extra payload copy rate: '
        f"{result['criteria']['managed_has_no_extra_payload_copy_rate']}")
    print(f"Managed bridge zero-copy: {result['bridge_zero_copy_pass']}")
    for delta in result['memcpy']['memory_total_deltas']:
        print(
            f"{delta['direction']}: count delta={delta['count_delta']}, "
            f"byte delta={delta['byte_delta']}")
    print('PASS' if result['pass'] else 'FAIL')
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
