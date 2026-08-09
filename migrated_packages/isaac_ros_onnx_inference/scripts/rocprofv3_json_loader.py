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
Strict ROCprofiler-v3 JSON loader used by the AMD copy audit.

The ROCprofiler JSON wrapper contains both activity records and lookup tables.
This module intentionally reads only the activity sections needed by the audit:
``buffer_records.memory_copy`` and ``buffer_records.kernel_dispatch``.  In
particular, strings, agents, and kernel symbols are metadata and never become
events merely because they look record-like.

The loader returns canonical events understood by ``copy_audit_common``.  The
canonical host/device labels are deliberately process-independent; the raw
agent handles are retained as evidence fields because ROCprofiler handles are
local to a process.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROCPROF_WRAPPER_KEY = 'rocprofiler-sdk-tool'
MEMORY_COPY_DOMAIN = 'memory_copy_trace'
KERNEL_TRACE_DOMAIN = 'kernel_trace'

_MEMORY_COPY_OPERATION_NUMBERS = {
    'MEMORY_COPY_NONE': 0,
    'MEMORY_COPY_HOST_TO_HOST': 1,
    'MEMORY_COPY_HOST_TO_DEVICE': 2,
    'MEMORY_COPY_DEVICE_TO_HOST': 3,
    'MEMORY_COPY_DEVICE_TO_DEVICE': 4,
}
_MEMORY_COPY_DIRECTIONS = {
    'MEMORY_COPY_HOST_TO_HOST': 'H2H',
    'MEMORY_COPY_HOST_TO_DEVICE': 'H2D',
    'MEMORY_COPY_DEVICE_TO_HOST': 'D2H',
    'MEMORY_COPY_DEVICE_TO_DEVICE': 'D2D',
}


class RocprofJsonError(ValueError):
    """Raised when a ROCprofiler JSON file is malformed as tooling input."""


class RocprofManifestError(ValueError):
    """Raised when a capture manifest is malformed as tooling input."""


class RocprofCapture:
    """Parsed events plus completeness and diagnostic evidence."""

    def __init__(self, paths: Optional[Sequence[str]] = None) -> None:
        self.events: List[Mapping[str, Any]] = []
        self.diagnostics: List[Dict[str, Any]] = []
        self.sections: Dict[str, Dict[str, Any]] = {}
        self.process_count = 0
        self.paths = list(paths or [])

    @property
    def memory_copy_events(self) -> List[Mapping[str, Any]]:
        return [event for event in self.events if event.get('rocprof_domain') == 'memory_copy']

    @property
    def kernel_events(self) -> List[Mapping[str, Any]]:
        return [event for event in self.events if event.get('rocprof_domain') == 'kernel_dispatch']


def _context(path: Path, process_index: int, domain: str, record_index: int) -> Dict[str, Any]:
    return {
        'path': str(path),
        'process_index': process_index,
        'domain': domain,
        'record_index': record_index,
    }


def _diagnostic(
        capture: RocprofCapture,
        diagnostic_kind: str,
        message: str,
        context: Mapping[str, Any],
        **details: Any) -> None:
    item: Dict[str, Any] = {
        'kind': diagnostic_kind, 'message': message, **dict(context)}
    item.update(details)
    capture.diagnostics.append(item)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer(value: Any) -> Optional[int]:
    if _is_integer(value):
        return value
    if isinstance(value, str) and re.fullmatch(r'[+-]?\d+', value.strip()):
        return int(value.strip())
    return None


def _key(value: Any) -> Optional[str]:
    number = _integer(value)
    if number is not None:
        return str(number)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RocprofJsonError(f'{label} must be a JSON object')
    return value


def _require_table(value: Any, label: str) -> Any:
    # ROCprofiler 7.1.1 emits arrays for agents, kernel_symbols, and
    # strings.buffer_records.  Accepting an object as well keeps the loader
    # compatible with the equivalent keyed representation used by some SDK
    # builds, while still rejecting scalars and nulls as schema errors.
    if not isinstance(value, (list, Mapping)):
        raise RocprofJsonError(f'{label} must be a JSON array or object')
    return value


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise RocprofJsonError(f'{path}: cannot read JSON: {exc}') from exc
    if not text.strip():
        raise RocprofJsonError(f'{path}: JSON report is empty')
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RocprofJsonError(f'{path}: invalid JSON: {exc}') from exc


def _table_entries(value: Any, key_field: str) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            if isinstance(raw_value, Mapping):
                item_key = _key(raw_value.get(key_field)) or _key(raw_key)
                if item_key is not None:
                    yield item_key, raw_value
        return
    for index, raw_value in enumerate(value):
        if isinstance(raw_value, Mapping):
            item_key = _key(raw_value.get(key_field)) or str(index)
            yield item_key, raw_value


def _build_agent_table(
        process: Mapping[str, Any],
        capture: RocprofCapture,
        context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_agents = _require_table(process['agents'], 'process.agents')
    agents: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_agents, Mapping):
        agent_entries = list(raw_agents.items())
    else:
        agent_entries = [(None, value) for value in raw_agents]
    for index, (raw_key, raw_agent) in enumerate(agent_entries):
        item_context = {**context, 'agent_index': index}
        if not isinstance(raw_agent, Mapping):
            _diagnostic(capture, 'malformed_agent_record',
                        'agent table entry is not an object', item_context)
            continue
        raw_id = raw_agent.get('id')
        if raw_id is None and raw_key is not None:
            raw_id = {'handle': raw_key}
        if not isinstance(raw_id, Mapping):
            _diagnostic(capture, 'agent_handle_unresolved',
                        'agent table entry has no id.handle object', item_context)
            continue
        handle = _parse_handle(raw_id.get('handle'))
        if handle is None:
            _diagnostic(capture, 'agent_handle_unresolved',
                        'agent table id.handle is not an integer', item_context,
                        value=raw_id.get('handle'))
            continue
        handle_key = str(handle)
        role = _agent_role(raw_agent)
        if role is None:
            _diagnostic(capture, 'agent_role_unresolved',
                        'agent table entry is neither a recognized CPU nor GPU agent',
                        item_context, handle=handle)
        if handle_key in agents:
            _diagnostic(capture, 'duplicate_agent_handle',
                        'agent handle is duplicated within one process', item_context,
                        handle=handle)
        agents[handle_key] = {
            'handle': handle,
            'role': role or 'unknown',
            'name': raw_agent.get('name', ''),
            'vendor_name': raw_agent.get('vendor_name', ''),
            'type': raw_agent.get('type'),
        }
    return agents


def _parse_handle(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        value = value.get('handle')
    return _integer(value)


def _agent_role(agent: Mapping[str, Any]) -> Optional[str]:
    agent_type = _integer(agent.get('type'))
    if agent_type == 1:
        return 'host'
    if agent_type == 2:
        return 'device'
    for field_name in ('vendor_name', 'product_name', 'name'):
        text = str(agent.get(field_name, '')).lower()
        if text in {'cpu', 'host'} or 'cpu' in text or 'host' in text:
            return 'host'
        if text in {'amd', 'gpu', 'device'} or 'gpu' in text or 'gfx' in text:
            return 'device'
    return None


def _build_operation_table(
        process: Mapping[str, Any],
        capture: RocprofCapture,
        context: Mapping[str, Any]) -> Dict[str, Any]:
    strings = _require_mapping(process['strings'], 'process.strings')
    if 'buffer_records' not in strings:
        raise RocprofJsonError('process.strings.buffer_records is missing')
    raw_records = _require_table(strings['buffer_records'], 'process.strings.buffer_records')
    operations: Dict[str, Any] = {}
    if isinstance(raw_records, Mapping):
        operation_entries = list(raw_records.items())
    else:
        operation_entries = [(None, value) for value in raw_records]
    for index, (raw_key, raw_entry) in enumerate(operation_entries):
        if not isinstance(raw_entry, Mapping):
            raise RocprofJsonError(
                f'process.strings.buffer_records[{index}] must be a JSON object')
        kind = _key(raw_entry.get('kind')) or _key(raw_key)
        if kind is None:
            raise RocprofJsonError(
                f'process.strings.buffer_records[{index}].kind is missing')
        raw_operations = raw_entry.get('operations')
        if not isinstance(raw_operations, (list, Mapping)):
            raise RocprofJsonError(
                f'process.strings.buffer_records[{index}].operations must be an array or object')
        operations[kind] = raw_operations
        # ROCprofiler 7.1.1 emits the symbolic kind name in this table (for
        # example MEMORY_COPY at list index 10), while activity records carry
        # the numeric enum value (10).  Keep both official representations as
        # lookup keys; the operation string itself remains the authority for
        # direction.
        operations.setdefault(str(index), raw_operations)
    return operations


def _lookup_operation(
        operation_table: Mapping[str, Any],
        kind: Any,
        operation: Any) -> Tuple[Optional[str], Optional[str]]:
    kind_key = _key(kind)
    if kind_key is None or kind_key not in operation_table:
        return None, 'buffer-record kind lookup failed'
    raw_operations = operation_table[kind_key]
    operation_index = _integer(operation)
    raw_name: Any = None
    if isinstance(raw_operations, Mapping):
        operation_key = _key(operation)
        if operation_key is not None:
            raw_name = raw_operations.get(operation_key)
        if raw_name is None and isinstance(operation, str):
            # A few JSON converters preserve the operation name instead of its
            # numeric enum.  It is safe to accept it only when it is present
            # in this kind's official operation table.
            if operation in raw_operations.values():
                raw_name = operation
    elif operation_index is not None and 0 <= operation_index < len(raw_operations):
        raw_name = raw_operations[operation_index]
    if not isinstance(raw_name, str) or not raw_name:
        return None, 'buffer-record operation lookup failed'
    return raw_name, None


def _direction(operation_name: Optional[str]) -> str:
    return _MEMORY_COPY_DIRECTIONS.get(str(operation_name), 'unknown')


def _schema_sanity_diagnostic(
        operation_name: Optional[str], operation: Any) -> Optional[str]:
    numeric = _integer(operation)
    expected = _MEMORY_COPY_OPERATION_NUMBERS.get(str(operation_name))
    if numeric is not None and expected is not None and numeric != expected:
        return (
            f'ROCprofiler memory-copy operation enum mismatch: numeric {numeric} '
            f'has metadata name {operation_name!r}, expected enum {expected}'
        )
    return None


def _parse_bytes(value: Any) -> Optional[int]:
    number = _integer(value)
    if number is not None and number >= 0:
        return number
    return None


def _symbol_table(
        process: Mapping[str, Any],
        capture: RocprofCapture,
        context: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    raw_symbols = _require_table(process['kernel_symbols'], 'process.kernel_symbols')
    symbols: Dict[str, Mapping[str, Any]] = {}
    for index, (symbol_key, raw_symbol) in enumerate(
            _table_entries(raw_symbols, 'kernel_id')):
        if not isinstance(raw_symbol, Mapping):
            _diagnostic(capture, 'malformed_kernel_symbol',
                        'kernel_symbols entry is not an object',
                        {**context, 'symbol_index': index})
            continue
        kernel_id = _key(raw_symbol.get('kernel_id')) or symbol_key
        if kernel_id is None:
            _diagnostic(capture, 'kernel_symbol_unresolved',
                        'kernel_symbols entry has no kernel_id',
                        {**context, 'symbol_index': index})
            continue
        symbols[kernel_id] = raw_symbol
    return symbols


def _kernel_name(symbol: Optional[Mapping[str, Any]]) -> Optional[str]:
    if symbol is None:
        return None
    for key in ('kernel_name', 'demangled_kernel_name', 'formatted_kernel_name'):
        value = symbol.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _event_base(
        domain: str,
        process_id: Any,
        record_index: int,
        raw_record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        'rocprof_domain': domain,
        'process_id': process_id,
        'rocprof_record_index': record_index,
        'raw_fields': sorted(str(key) for key in raw_record),
    }


def _parse_memory_copies(
        records: Any,
        path: Path,
        process_index: int,
        process_id: Any,
        agents: Mapping[str, Mapping[str, Any]],
        operation_table: Mapping[str, Any],
        capture: RocprofCapture) -> List[Mapping[str, Any]]:
    events: List[Mapping[str, Any]] = []
    if records is None:
        return events
    for index, raw_record in enumerate(records):
        context = _context(path, process_index, 'memory_copy', index)
        if not isinstance(raw_record, Mapping):
            _diagnostic(capture, 'malformed_memory_copy_record',
                        'memory_copy entry is not an object', context)
            events.append({
                **_event_base('memory_copy', process_id, index, {}),
                'kind': 'memory_copy',
                'operation': '<malformed memory_copy record>',
                'direction': 'unknown',
                'bytes': None,
                'source_agent': 'unknown',
                'destination_agent': 'unknown',
            })
            continue

        missing = [key for key in ('kind', 'operation', 'src_agent_id',
                                   'dst_agent_id', 'bytes') if key not in raw_record]
        if missing:
            _diagnostic(capture, 'incomplete_memory_copy_record',
                        'memory_copy record is missing required fields', context,
                        missing_fields=missing)

        kind = raw_record.get('kind')
        operation = raw_record.get('operation')
        operation_name, lookup_error = _lookup_operation(operation_table, kind, operation)
        if lookup_error:
            _diagnostic(capture, 'operation_lookup_failed', lookup_error, context,
                        record_kind=kind, operation=operation)
        sanity_error = _schema_sanity_diagnostic(operation_name, operation)
        if sanity_error:
            _diagnostic(capture, 'schema_version_mismatch', sanity_error, context,
                        record_kind=kind, operation=operation,
                        operation_name=operation_name)

        source_handle = _parse_handle(raw_record.get('src_agent_id'))
        destination_handle = _parse_handle(raw_record.get('dst_agent_id'))
        source_info = agents.get(str(source_handle)) if source_handle is not None else None
        destination_info = (
            agents.get(str(destination_handle)) if destination_handle is not None else None)
        if source_handle is None or source_info is None:
            _diagnostic(capture, 'agent_handle_unresolved',
                        'source agent handle cannot be resolved in this process', context,
                        side='source', value=raw_record.get('src_agent_id'))
        if destination_handle is None or destination_info is None:
            _diagnostic(capture, 'agent_handle_unresolved',
                        'destination agent handle cannot be resolved in this process', context,
                        side='destination', value=raw_record.get('dst_agent_id'))

        source_role = source_info['role'] if source_info else 'unknown'
        destination_role = destination_info['role'] if destination_info else 'unknown'
        direction = _direction(operation_name)
        if direction == 'unknown':
            _diagnostic(capture, 'unknown_memory_copy_direction',
                        'memory-copy operation does not identify a canonical direction',
                        context, operation_name=operation_name)
        event = {
            **_event_base('memory_copy', process_id, index, raw_record),
            'kind': 'memory_copy',
            'operation': operation_name or f'<unknown operation {operation!r}>',
            'direction': direction,
            'bytes': _parse_bytes(raw_record.get('bytes')),
            'source_agent': source_role,
            'destination_agent': destination_role,
            'source_agent_handle': source_handle,
            'destination_agent_handle': destination_handle,
            'rocprof_kind': kind,
            'rocprof_operation': operation,
        }
        events.append(event)
    return events


def _parse_kernel_dispatches(
        records: Any,
        path: Path,
        process_index: int,
        process_id: Any,
        agents: Mapping[str, Mapping[str, Any]],
        operation_table: Mapping[str, Any],
        symbols: Mapping[str, Mapping[str, Any]],
        capture: RocprofCapture) -> List[Mapping[str, Any]]:
    events: List[Mapping[str, Any]] = []
    if records is None:
        return events
    for index, raw_record in enumerate(records):
        context = _context(path, process_index, 'kernel_dispatch', index)
        if not isinstance(raw_record, Mapping):
            _diagnostic(capture, 'malformed_kernel_dispatch_record',
                        'kernel_dispatch entry is not an object', context)
            events.append({
                **_event_base('kernel_dispatch', process_id, index, {}),
                'kind': 'kernel_dispatch',
                'kernel_name': '<malformed kernel_dispatch record>',
                'operation': '<malformed kernel_dispatch record>',
            })
            continue

        missing = [key for key in ('kind', 'operation', 'dispatch_info')
                   if key not in raw_record]
        if missing:
            _diagnostic(capture, 'incomplete_kernel_dispatch_record',
                        'kernel_dispatch record is missing required fields', context,
                        missing_fields=missing)
        kind = raw_record.get('kind')
        operation = raw_record.get('operation')
        operation_name, lookup_error = _lookup_operation(operation_table, kind, operation)
        if lookup_error:
            _diagnostic(capture, 'operation_lookup_failed', lookup_error, context,
                        record_kind=kind, operation=operation)

        dispatch_info = raw_record.get('dispatch_info')
        if not isinstance(dispatch_info, Mapping):
            _diagnostic(capture, 'incomplete_kernel_dispatch_record',
                        'kernel_dispatch.dispatch_info is not an object', context)
            dispatch_info = {}
        kernel_id = _integer(dispatch_info.get('kernel_id'))
        if kernel_id is None:
            _diagnostic(capture, 'kernel_id_unresolved',
                        'kernel dispatch has no integer dispatch_info.kernel_id', context,
                        value=dispatch_info.get('kernel_id'))
        symbol = symbols.get(str(kernel_id)) if kernel_id is not None else None
        name = _kernel_name(symbol)
        if symbol is None or name is None:
            _diagnostic(capture, 'kernel_id_unresolved',
                        'kernel dispatch kernel_id is absent from kernel_symbols', context,
                        kernel_id=kernel_id)
        dispatch_agent_handle = _parse_handle(dispatch_info.get('agent_id'))
        dispatch_agent = agents.get(str(dispatch_agent_handle)) \
            if dispatch_agent_handle is not None else None
        if dispatch_agent_handle is None or dispatch_agent is None:
            _diagnostic(capture, 'agent_handle_unresolved',
                        'kernel dispatch agent handle cannot be resolved in this process',
                        context, side='dispatch', value=dispatch_info.get('agent_id'))
        event_name = name or f'<unknown kernel {kernel_id!r}>'
        events.append({
            **_event_base('kernel_dispatch', process_id, index, raw_record),
            'kind': 'kernel_dispatch',
            'kernel_name': event_name,
            'operation': event_name,
            'kernel_id': kernel_id,
            'kernel_agent_handle': dispatch_agent_handle,
            'kernel_agent': dispatch_agent['role'] if dispatch_agent else 'unknown',
            'rocprof_kind': kind,
            'rocprof_operation': operation,
            'kernel_operation_name': operation_name,
        })
    return events


def _section_state(
        process_states: Sequence[Mapping[str, Any]],
        section: str) -> Dict[str, Any]:
    states = [dict(item) for item in process_states]
    present = [bool(item.get('present')) for item in states]
    count = sum(int(item.get('record_count', 0)) for item in states)
    all_present = bool(states) and all(present)
    return {
        'section': section,
        'present': all_present,
        'record_count': count,
        'processes': states,
        'status': 'complete' if all_present and count > 0 else 'incomplete',
    }


def load_rocprof_json(paths: Sequence[Path]) -> RocprofCapture:
    """
    Parse official ROCprofiler JSON activity sections from ``paths``.

    Structural errors raise :class:`RocprofJsonError` and are tooling errors.
    Record-level errors are retained in ``capture.diagnostics`` and malformed
    records remain represented by an unknown event so an analyzer can only
    conclude ``INCONCLUSIVE``.
    """
    capture = RocprofCapture(paths=[str(Path(path)) for path in paths])
    process_section_states: Dict[str, List[Dict[str, Any]]] = {
        'memory_copy': [],
        'kernel_dispatch': [],
    }
    for raw_path in paths:
        path = Path(raw_path)
        document = _read_json(path)
        if not isinstance(document, Mapping):
            raise RocprofJsonError(f'{path}: top-level JSON must be an object')
        if ROCPROF_WRAPPER_KEY not in document:
            raise RocprofJsonError(
                f'{path}: top-level {ROCPROF_WRAPPER_KEY!r} wrapper is missing')
        processes = document[ROCPROF_WRAPPER_KEY]
        if not isinstance(processes, list):
            raise RocprofJsonError(
                f'{path}: {ROCPROF_WRAPPER_KEY} must be a JSON array')
        for process_index, process in enumerate(processes):
            if not isinstance(process, Mapping):
                raise RocprofJsonError(
                    f'{path}: process entry {process_index} must be a JSON object')
            for key in ('agents', 'strings', 'kernel_symbols', 'buffer_records'):
                if key not in process:
                    raise RocprofJsonError(
                        f'{path}: process entry {process_index}.{key} is missing')
            buffer_records = _require_mapping(
                process['buffer_records'],
                f'{path}: process entry {process_index}.buffer_records')
            context = {
                'path': str(path),
                'process_index': process_index,
            }
            metadata = process.get('metadata', {})
            if metadata is not None and not isinstance(metadata, Mapping):
                raise RocprofJsonError(
                    f'{path}: process entry {process_index}.metadata must be an object')
            process_id = metadata.get('pid', f'process-{process_index}') \
                if isinstance(metadata, Mapping) else f'process-{process_index}'
            agents = _build_agent_table(process, capture, context)
            operation_table = _build_operation_table(process, capture, context)
            symbols = _symbol_table(process, capture, context)

            lane_events: List[Mapping[str, Any]] = []
            for section, domain in (('memory_copy', 'memory_copy'),
                                    ('kernel_dispatch', 'kernel_dispatch')):
                if section not in buffer_records:
                    process_section_states[section].append({
                        'process_index': process_index,
                        'pid': process_id,
                        'present': False,
                        'record_count': 0,
                    })
                    continue
                records = buffer_records[section]
                if not isinstance(records, list):
                    raise RocprofJsonError(
                        f'{path}: process entry {process_index}.buffer_records.{section} '
                        'must be a JSON array')
                process_section_states[section].append({
                    'process_index': process_index,
                    'pid': process_id,
                    'present': True,
                    'record_count': len(records),
                })
                if section == 'memory_copy':
                    lane_events.extend(_parse_memory_copies(
                        records, path, process_index, process_id, agents,
                        operation_table, capture))
                else:
                    lane_events.extend(_parse_kernel_dispatches(
                        records, path, process_index, process_id, agents,
                        operation_table, symbols, capture))
            capture.events.extend(lane_events)
            capture.process_count += 1

    capture.sections = {
        MEMORY_COPY_DOMAIN: _section_state(
            process_section_states['memory_copy'], 'memory_copy'),
        KERNEL_TRACE_DOMAIN: _section_state(
            process_section_states['kernel_dispatch'], 'kernel_dispatch'),
    }
    return capture


def load_manifest(path: Path) -> Mapping[str, Any]:
    """Load and validate a capture manifest written by the AMD runner."""
    document = _read_json(Path(path))
    if not isinstance(document, Mapping):
        raise RocprofManifestError(f'{path}: manifest must be a JSON object')
    for key in ('tracing_domains', 'output_formats'):
        value = document.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise RocprofManifestError(
                f'{path}: manifest.{key} must be an array of strings')
    return document


def manifest_domain_status(
        capture: RocprofCapture,
        manifest: Optional[Mapping[str, Any]],
        domain: str) -> Dict[str, Any]:
    """Describe whether a requested ROCprofiler domain was captured."""
    if manifest is None:
        return {
            'domain': domain,
            'status': 'not_checked',
            'requested': None,
            'record_count': capture.sections.get(domain, {}).get('record_count', 0),
        }
    requested = domain in set(manifest.get('tracing_domains', []))
    section = capture.sections.get(domain, {})
    count = int(section.get('record_count', 0))
    if not requested:
        status = 'not_requested'
    elif not section.get('present') or count == 0:
        status = 'copy_domain_incomplete' if domain == MEMORY_COPY_DOMAIN \
            else 'kernel_domain_incomplete'
    else:
        status = 'complete'
    return {
        'domain': domain,
        'status': status,
        'requested': requested,
        'record_count': count,
        'section_present': bool(section.get('present')),
        'processes': section.get('processes', []),
    }


def capture_completeness(
        capture: RocprofCapture,
        manifest: Optional[Mapping[str, Any]],
        *,
        require_kernel_evidence: bool = True) -> Dict[str, Any]:
    """Return manifest-aware completeness without deciding PASS/FAIL."""
    memory = manifest_domain_status(capture, manifest, MEMORY_COPY_DOMAIN)
    kernel = manifest_domain_status(capture, manifest, KERNEL_TRACE_DOMAIN)
    memory_complete = memory['status'] in {'complete', 'not_checked'}
    kernel_complete = kernel['status'] in {'complete', 'not_checked'}
    return {
        'memory_copy': memory,
        'kernel': kernel,
        'parser_diagnostic_count': len(capture.diagnostics),
        'parser_diagnostics': capture.diagnostics,
        'memory_copy_complete': memory_complete and not any(
            item.get('domain') == 'memory_copy' for item in capture.diagnostics),
        'kernel_complete': kernel_complete and not any(
            item.get('domain') == 'kernel_dispatch' for item in capture.diagnostics),
        'kernel_evidence_required': require_kernel_evidence,
    }


def _csv_field(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower().replace(' ', '_'): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace(' ', '_')
        if key in lowered:
            return lowered[key]
    return None


def _csv_direction(operation: Any) -> str:
    text = str(operation or '').upper().replace(' ', '_')
    if text in _MEMORY_COPY_DIRECTIONS:
        return _MEMORY_COPY_DIRECTIONS[text]
    compact = re.sub(r'[^A-Z0-9]', '', text)
    if 'HOSTTODEVICE' in compact or 'HTOD' in compact:
        return 'H2D'
    if 'DEVICETOHOST' in compact or 'DTOH' in compact:
        return 'D2H'
    if 'DEVICETODEVICE' in compact or 'DTOD' in compact:
        return 'D2D'
    if 'HOSTTOHOST' in compact or 'HTOH' in compact:
        return 'H2H'
    return 'unknown'


def summarize_csv(paths: Sequence[Path]) -> Dict[str, Any]:
    """
    Summarize CSV only for cross-checking JSON direction/agent/count.

    No byte total from this function is used as audit evidence.  JSON remains
    the authoritative source for byte counts and all closure decisions.
    """
    copies: Counter = Counter()
    kernels: Counter = Counter()
    rows = 0
    for raw_path in paths:
        path = Path(raw_path)
        try:
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    rows += 1
                    operation = _csv_field(row, 'Operation', 'operation')
                    kernel_name = _csv_field(row, 'Kernel_Name', 'Kernel', 'Name')
                    source = _csv_field(
                        row, 'Source_Agent_Id', 'Src_Agent_Id', 'Source Agent Id',
                        'Source_Agent')
                    destination = _csv_field(
                        row, 'Destination_Agent_Id', 'Dst_Agent_Id', 'Destination Agent Id',
                        'Destination_Agent')
                    name = str(kernel_name or operation or '')
                    if 'kernel' in path.name.lower() or kernel_name is not None:
                        kernels[name] += 1
                    elif 'memory' in path.name.lower() or operation is not None:
                        copies[(_csv_direction(operation), str(source or ''),
                                str(destination or ''))] += 1
        except (OSError, csv.Error) as exc:
            raise RocprofJsonError(f'{path}: cannot read CSV: {exc}') from exc
    return {
        'row_count': rows,
        'memory_copy': [
            {'direction': key[0], 'source_agent': key[1],
             'destination_agent': key[2], 'count': copies[key]}
            for key in sorted(copies, key=str)
        ],
        'kernels': [{'name': key, 'count': kernels[key]}
                    for key in sorted(kernels)],
    }


def cross_check_csv(
        capture: RocprofCapture,
        csv_paths: Sequence[Path]) -> Dict[str, Any]:
    """Compare CSV direction/agent/count with JSON without trusting CSV bytes."""
    csv_summary = summarize_csv(csv_paths)
    json_counts: Counter = Counter()
    for event in capture.memory_copy_events:
        json_counts[(event.get('direction', 'unknown'),
                     str(event.get('source_agent_handle', '')),
                     str(event.get('destination_agent_handle', '')))] += 1
    csv_counts = Counter((item['direction'], item['source_agent'],
                          item['destination_agent'])
                         for item in csv_summary['memory_copy'])
    deltas = [
        {'key': list(key), 'json_count': json_counts[key], 'csv_count': csv_counts[key]}
        for key in sorted(set(json_counts) | set(csv_counts), key=str)
        if json_counts[key] != csv_counts[key]
    ]
    return {
        'status': 'PASS' if not deltas else 'INCONCLUSIVE',
        'json_authoritative': True,
        'direction_agent_count_deltas': deltas,
        'json_memory_copy_count': sum(json_counts.values()),
        'csv_memory_copy_count': sum(csv_counts.values()),
        'csv': csv_summary,
    }


# Small aliases make the internal module convenient for offline notebooks and
# preserve a discoverable name if the command is moved into another package.
load_capture = load_rocprof_json
