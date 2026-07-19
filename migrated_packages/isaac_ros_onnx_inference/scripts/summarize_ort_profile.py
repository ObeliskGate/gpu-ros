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

"""Summarize ONNX Runtime execution-provider assignments from profile files."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

CPU_PROVIDER = 'CPUExecutionProvider'


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Report the providers that executed ONNX Runtime node kernels and '
            'list any nodes assigned to CPUExecutionProvider.'))
    parser.add_argument('profiles', nargs='+', type=Path, help='ORT profile JSON files')
    parser.add_argument(
        '--expected-provider',
        help='Fail if this provider has no recorded kernel events')
    parser.add_argument(
        '--require-no-cpu-nodes',
        action='store_true',
        help='Fail if CPUExecutionProvider kernel events are present')
    parser.add_argument(
        '--max-node-names',
        type=int,
        default=50,
        help='Maximum unique node names printed per provider (default: 50)')
    parser.add_argument('--output-json', type=Path, help='Optional machine-readable report')
    return parser.parse_args()


def event_node_name(event):
    """Return a stable node label for a kernel event."""
    args = event.get('args') or {}
    name = str(args.get('node_name') or event.get('name') or '<unnamed>')
    if name.endswith('_kernel_time'):
        name = name[:-len('_kernel_time')]
    op_name = args.get('op_name')
    return f'{name} ({op_name})' if op_name else name


def summarize_profile(path):
    """Collect provider counts and node names from one ORT profile."""
    with path.open(encoding='utf-8') as profile_file:
        events = json.load(profile_file)
    if not isinstance(events, list):
        raise ValueError('top-level JSON value must be an array')

    providers = defaultdict(lambda: {'kernel_events': 0, 'duration_us': 0.0, 'nodes': set()})
    for event in events:
        if not isinstance(event, dict):
            continue
        event_args = event.get('args') or {}
        provider = event_args.get('provider')
        if not provider:
            continue

        provider_data = providers[str(provider)]
        provider_data['kernel_events'] += 1
        provider_data['duration_us'] += float(event.get('dur') or 0.0)
        provider_data['nodes'].add(event_node_name(event))

    provider_report = {
        provider: {
            'kernel_events': data['kernel_events'],
            'duration_us': data['duration_us'],
            'unique_nodes': sorted(data['nodes']),
        }
        for provider, data in sorted(providers.items())
    }
    return {
        'path': str(path),
        'provider_kernel_events_found': bool(provider_report),
        'cpu_fallback_observed': CPU_PROVIDER in provider_report,
        'providers': provider_report,
    }


def print_report(report, max_node_names):
    """Print one profile summary."""
    print(f"Profile: {report['path']}")
    providers = report['providers']
    if not providers:
        print('  Provider assignment: unknown (no provider kernel events found)')
        return

    print('  Provider assignment:')
    for provider, data in providers.items():
        print(
            f"    {provider}: {len(data['unique_nodes'])} unique nodes, "
            f"{data['kernel_events']} kernel events, {data['duration_us']:.3f} us")

    cpu_nodes = providers.get(CPU_PROVIDER, {}).get('unique_nodes', [])
    if not cpu_nodes:
        print('  CPU fallback: none observed')
        return

    print(f'  CPU fallback: detected ({len(cpu_nodes)} unique nodes)')
    for node_name in cpu_nodes[:max_node_names]:
        print(f'    - {node_name}')
    omitted = len(cpu_nodes) - max_node_names
    if omitted > 0:
        print(f'    ... {omitted} more nodes omitted')


def main():
    """Run the provider-assignment report."""
    args = parse_args()
    reports = []
    invalid_profile = False
    cpu_nodes_found = False
    expected_provider_missing = False

    for path in args.profiles:
        try:
            report = summarize_profile(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'ERROR: cannot read profile {path}: {exc}', file=sys.stderr)
            invalid_profile = True
            continue

        reports.append(report)
        print_report(report, args.max_node_names)
        providers = report['providers']
        if not providers:
            invalid_profile = True
        if CPU_PROVIDER in providers:
            cpu_nodes_found = True
        if args.expected_provider and args.expected_provider not in providers:
            print(
                f"ERROR: expected provider '{args.expected_provider}' has no kernel "
                f'events in {path}',
                file=sys.stderr)
            expected_provider_missing = True

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({'profiles': reports}, indent=2) + '\n')

    if invalid_profile or expected_provider_missing:
        return 2
    if args.require_no_cpu_nodes and cpu_nodes_found:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
