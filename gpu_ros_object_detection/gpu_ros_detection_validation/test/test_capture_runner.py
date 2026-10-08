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

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

MATRIX_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2b_benchmark_matrix.sh'
MATRIX_GRAPH_NAMES = {
    'yolov8': (
        'gpu_ros_yolov8_phase2a_amd_graph.py',
        'gpu_ros_yolov8_phase2b_amd_staged_control_graph.py',
        'gpu_ros_yolov8_phase2b_amd_managed_graph.py',
    ),
    'rtdetr': (
        'gpu_ros_rtdetr_phase2a_amd_graph.py',
        'gpu_ros_rtdetr_phase2b_amd_staged_control_graph.py',
        'gpu_ros_rtdetr_phase2b_amd_managed_graph.py',
    ),
}
MATRIX_MODEL_PATHS = {
    'yolov8': Path('models/yolov8/yolov8s.onnx'),
    'rtdetr': Path('models/rtdetrv2_r50/rtdetrv2_r50.onnx'),
}


def _git(path, *arguments):
    return subprocess.run(
        ['git', '-C', str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _create_matrix_workspace(path):
    path.mkdir(parents=True)
    benchmark_dir = path / 'gpu_ros_object_detection' / 'benchmarks'
    benchmark_dir.mkdir(parents=True)
    for graph in sorted({graph for graphs in MATRIX_GRAPH_NAMES.values() for graph in graphs}):
        (benchmark_dir / graph).write_text('# matrix fixture graph\n')
    (path / 'tracked.txt').write_text('committed fixture\n')
    _git(path, 'init', '-b', 'main')
    _git(path, 'config', 'user.name', 'Matrix Test')
    _git(path, 'config', 'user.email', 'matrix-test@example.invalid')
    _git(path, 'add', '--all')
    _git(path, 'commit', '-m', 'matrix fixture')
    return path


def _create_assets(root, model, dataset_contents=b'fixture dataset\n'):
    model_path = root / MATRIX_MODEL_PATHS[model]
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(f'fixture model: {model}\n'.encode())
    dataset_path = root / 'datasets' / 'r2bdataset2024_v1' / 'r2b_robotarm'
    dataset_path.mkdir(parents=True)
    (dataset_path / 'sample.mcap').write_bytes(dataset_contents)
    return model_path, dataset_path


def _install_fake_launch_test(root):
    bin_dir = root / 'fake-bin'
    bin_dir.mkdir()
    launch_test = bin_dir / 'launch_test'
    launch_test.write_text(
        '#!/usr/bin/env bash\n'
        'set -eu\n'
        'printf "%s\\t%s\\n" "${1##*/}" '
        '"${ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT:-}" >>"${LAUNCH_LOG}"\n'
        'printf \'{"fixture":true}\\n\' >"${R2B_RESULT_FILE}"\n'
    )
    launch_test.chmod(launch_test.stat().st_mode | 0o111)
    return bin_dir


def _run_matrix(
    root,
    workspace,
    bin_dir,
    model,
    matrix_name,
    asset_environment,
    extra_environment=None,
):
    launch_log = root / f'{matrix_name}-launches.tsv'
    environment = os.environ.copy()
    for name in (
        'CAPTURE_MODEL_PATH',
        'CAPTURE_INPUT_BAG',
        'ROS2_BENCHMARK_OVERRIDE_INPUT_DATA_PATH',
        'OVG_ASSETS_ROOT',
        'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT',
        'ONNXRUNTIME_ROOT',
        'ONNXRUNTIME_LIBRARY',
    ):
        environment.pop(name, None)
    environment.update(
        {
            'OVG_WORKSPACE_ROOT': str(workspace),
            'OVG_RESULTS_ROOT': str(root / 'results'),
            'ROS_DISTRO': 'matrix-fixture-no-ros',
            'PATH': f'{bin_dir}{os.pathsep}{environment.get("PATH", "")}',
            'LAUNCH_LOG': str(launch_log),
        }
    )
    environment.update(asset_environment)
    if extra_environment:
        environment.update(extra_environment)
    result = subprocess.run(
        [str(MATRIX_SCRIPT_PATH), model, matrix_name],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output_dir = root / 'results' / 'phase2b-benchmark-matrix' / matrix_name
    return result, output_dir, launch_log


def _read_matrix_manifest(output_dir):
    return {
        key: value
        for key, _, value in (
            line.partition('=')
            for line in (output_dir / 'matrix_manifest.txt').read_text().splitlines()
        )
    }


def _expected_relative_tree_sha256(dataset_path):
    files = sorted(
        (path for path in dataset_path.rglob('*') if path.is_file()),
        key=lambda path: path.relative_to(dataset_path).as_posix(),
    )
    listing = ''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
        f'./{path.relative_to(dataset_path).as_posix()}\n'
        for path in files
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def _assert_matrix_invocation(result, output_dir, launch_log, model, assets_root):
    assert result.returncode == 0, f'{result.stdout}\n{result.stderr}'
    records = [line.split('\t') for line in launch_log.read_text().splitlines()]
    expected_indices = (0, 1, 2, 1, 2, 0, 2, 0, 1)
    assert len(records) == 9
    assert [record[0] for record in records] == [
        MATRIX_GRAPH_NAMES[model][index] for index in expected_indices
    ]
    assert all(len(record) == 2 for record in records)
    assert {record[1] for record in records} == {str(assets_root)}
    manifest = _read_matrix_manifest(output_dir)
    model_path = assets_root / MATRIX_MODEL_PATHS[model]
    dataset_path = assets_root / 'datasets' / 'r2bdataset2024_v1' / 'r2b_robotarm'
    assert manifest['schema_version'] == '3'
    assert manifest['assets_root'] == str(assets_root)
    assert manifest['model_path'] == str(model_path)
    assert manifest['model_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert manifest['dataset_path'] == str(dataset_path)
    assert manifest['dataset_tree_hash_algorithm'] == 'sha256sum-sorted-relative-path-v1'
    for result_file in output_dir.glob('round-*.json'):
        assert json.loads(result_file.read_text()) == {'fixture': True}
    assert len(list(output_dir.glob('round-*.json'))) == 9
    return manifest


SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_fixed_input_capture.sh'
AMD_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2a_fixed_input_capture.sh'
AUDIT_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_yolov8_transport_audit.sh'


_NVIDIA_ROS2_FIXTURE_SOURCE = r'''
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _write_env_path(name, value):
    Path(os.environ[name]).write_text(f'{value}\n')


def _exit_on_signal(_signum, _frame):
    raise SystemExit(0)


def _wait_for_ready(name):
    path = Path(os.environ[name])
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f'fixture child did not become ready: {name}')
        time.sleep(0.01)


def _child(role, mode):
    received = 0

    def _count_signal(_signum, _frame):
        nonlocal received
        received += 1
        _write_env_path('CAPTURE_FIXTURE_COMPONENT_SIGNAL_FILE', received)

    if role == 'component' and mode == 'kill':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif role == 'component' and mode == 'term':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _exit_on_signal)
    elif role == 'component' and mode == 'relay_once':
        signal.signal(signal.SIGINT, _count_signal)
        signal.signal(signal.SIGTERM, _exit_on_signal)
    else:
        signal.signal(signal.SIGINT, _exit_on_signal)
        signal.signal(signal.SIGTERM, _exit_on_signal)
    if role == 'component':
        _write_env_path('CAPTURE_FIXTURE_COMPONENT_PID_FILE', os.getpid())
        _write_env_path('CAPTURE_FIXTURE_COMPONENT_READY_FILE', 'ready')
    else:
        _write_env_path('CAPTURE_FIXTURE_CONTAINER_PID_FILE', os.getpid())
        _write_env_path('CAPTURE_FIXTURE_CONTAINER_READY_FILE', 'ready')
    if role == 'component' and mode == 'relay_once':
        while received == 0:
            signal.pause()
        # Keep teardown alive long enough to observe launch's forwarded signal.
        time.sleep(0.25)
        raise SystemExit(0 if received == 1 else 7)
    while True:
        signal.pause()


def _topic_info(topic):
    started = Path(os.environ['CAPTURE_FIXTURE_STARTED_PATH']).exists()
    is_detection_topic = topic.endswith('/detections_output')
    publisher = int(started and is_detection_topic)
    record_ready = Path(os.environ['CAPTURE_FIXTURE_RECORD_READY_FILE']).exists()
    subscription = int(started and (not is_detection_topic or record_ready))
    print(f'Publisher count: {publisher}')
    print(f'Subscription count: {subscription}')
    return 0


def _launch():
    mode = os.environ['CAPTURE_FIXTURE_MODE']

    children = []

    def _on_signal(signum, _frame):
        if mode == 'relay_once':
            receipt = Path(os.environ['CAPTURE_FIXTURE_COMPONENT_SIGNAL_FILE'])
            deadline = time.monotonic() + 0.1
            while not receipt.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
        for child in children:
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    container = subprocess.Popen(
        [sys.executable, __file__, 'child', 'container', mode],
    )
    component_command = (
        ['/bin/sh', '-c', 'exit 7']
        if mode == 'crash'
        else [sys.executable, __file__, 'child', 'component', mode]
    )
    component = subprocess.Popen(component_command)
    children.extend((container, component))
    _wait_for_ready('CAPTURE_FIXTURE_CONTAINER_READY_FILE')
    if mode != 'crash':
        _wait_for_ready('CAPTURE_FIXTURE_COMPONENT_READY_FILE')
    _write_env_path('CAPTURE_FIXTURE_LAUNCH_PID_FILE', os.getpid())
    print(
        f'[INFO] [component_container-1]: process started with pid [{container.pid}]',
        flush=True,
    )
    print(
        f'[INFO] [component_container_mt-2]: process started with pid [{component.pid}]',
        flush=True,
    )
    Path(os.environ['CAPTURE_FIXTURE_STARTED_PATH']).write_text('started\n')

    component_status = component.wait()
    if component_status == 0:
        if mode == 'contradictory':
            print(
                '[INFO] [component_container_mt-2]: process has finished '
                f'cleanly [pid {component.pid + 1}]',
                flush=True,
            )
        elif mode != 'missing':
            print(
                '[INFO] [component_container_mt-2]: process has finished '
                f'cleanly [pid {component.pid}]',
                flush=True,
            )
    else:
        print(
            '[ERROR] [component_container_mt-2]: process has died '
            f"[pid {component.pid}, exit code {component_status}, "
            "cmd '/bin/sh -c exit 7'].",
            flush=True,
        )

    container_status = container.wait()
    if container_status == 0:
        print(
            '[INFO] [component_container-1]: process has finished '
            f'cleanly [pid {container.pid}]',
            flush=True,
        )
    else:
        print(
            '[ERROR] [component_container-1]: process has died '
            f'[pid {container.pid}, exit code {container_status}, cmd '
            "'fixture container'].",
            flush=True,
        )
    _write_env_path('CAPTURE_FIXTURE_LAUNCH_STATUS_FILE', 0)
    return 0


def _record():
    mode = os.environ['CAPTURE_FIXTURE_RECORD_MODE']
    if mode == 'kill':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif mode == 'term':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _exit_on_signal)
    elif mode == 'inherited':
        # Like Python ROS CLI entry points, retain the inherited SIGINT policy.
        signal.signal(signal.SIGTERM, _exit_on_signal)
    else:
        signal.signal(signal.SIGINT, _exit_on_signal)
        signal.signal(signal.SIGTERM, _exit_on_signal)
    output_path = Path(sys.argv[sys.argv.index('--output') + 1])
    output_path.mkdir(parents=True, exist_ok=True)
    _write_env_path('CAPTURE_FIXTURE_RECORD_PID_FILE', os.getpid())
    _write_env_path('CAPTURE_FIXTURE_RECORD_READY_FILE', 'ready')
    while True:
        signal.pause()

def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == 'child':
        _child(arguments[1], arguments[2])
        return 0
    if arguments[:2] == ['topic', 'info']:
        return _topic_info(arguments[2])
    if arguments[:1] == ['launch']:
        return _launch()
    if arguments[:2] == ['bag', 'record']:
        return _record()
    if arguments[:2] == ['bag', 'play']:
        return int(os.environ['CAPTURE_FIXTURE_PLAY_STATUS'])
    if arguments[:2] == ['bag', 'info']:
        print('Messages: 1')
        return 0
    print(f'unexpected ros2 fixture arguments: {arguments}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
'''


def _create_nvidia_capture_fixture(
    root, mode, bag_play_status, record_mode, stop_grace_seconds, stop_term_seconds
):
    assets_root = root / 'assets'
    model_path = assets_root / 'models' / 'synthetica_detr_v1.0.0_onnx' / 'sdetr_grasp.onnx'
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b'fixture model\n')
    yolov8_model_path = assets_root / 'models' / 'yolov8' / 'yolov8s.onnx'
    yolov8_model_path.parent.mkdir(parents=True)
    yolov8_model_path.write_bytes(b'fixture YOLOv8 model\n')
    input_bag = assets_root / 'datasets' / 'r2bdataset2024_v1' / 'r2b_robotarm'
    input_bag.mkdir(parents=True)
    (input_bag / 'metadata.yaml').write_text('fixture bag metadata\n')

    workspace = root / 'workspace'
    workspace.mkdir()
    bin_dir = root / 'fake-bin'
    bin_dir.mkdir()
    ros2_path = bin_dir / 'ros2'
    ros2_path.write_text(f'#!{sys.executable}\n' + _NVIDIA_ROS2_FIXTURE_SOURCE)
    ros2_path.chmod(ros2_path.stat().st_mode | 0o111)
    profiler_path = bin_dir / 'nsys'
    profiler_path.write_text(
        f'#!{sys.executable}\n'
        'import signal, subprocess, sys\n'
        'child = subprocess.Popen(sys.argv[sys.argv.index("bash"):], start_new_session=True)\n'
        'def forward(signum, _frame):\n'
        '    if child.poll() is None:\n'
        '        child.send_signal(signum)\n'
        'signal.signal(signal.SIGINT, forward)\n'
        'signal.signal(signal.SIGTERM, forward)\n'
        'raise SystemExit(child.wait())\n'
    )
    profiler_path.chmod(profiler_path.stat().st_mode | 0o111)

    output_root = root / 'capture-output'
    fixture_paths = {
        'CAPTURE_FIXTURE_STARTED_PATH': root / 'launch-started',
        'CAPTURE_FIXTURE_LAUNCH_PID_FILE': root / 'launch-pid',
        'CAPTURE_FIXTURE_COMPONENT_PID_FILE': root / 'component-pid',
        'CAPTURE_FIXTURE_CONTAINER_PID_FILE': root / 'container-pid',
        'CAPTURE_FIXTURE_RECORD_PID_FILE': root / 'record-pid',
        'CAPTURE_FIXTURE_CONTAINER_READY_FILE': root / 'container-ready',
        'CAPTURE_FIXTURE_COMPONENT_READY_FILE': root / 'component-ready',
        'CAPTURE_FIXTURE_COMPONENT_SIGNAL_FILE': root / 'component-signal-count',
        'CAPTURE_FIXTURE_RECORD_READY_FILE': root / 'record-ready',
        'CAPTURE_FIXTURE_LAUNCH_STATUS_FILE': root / 'launch-status',
    }
    environment = os.environ.copy()
    for name in (
        'CAPTURE_ORT_PROFILE_PREFIX',
        'CAPTURE_BINDING_REPORT_PATH',
        'CAPTURE_NSYS_OUTPUT',
        'CAPTURE_MODEL_PATH',
        'CAPTURE_INPUT_BAG',
        'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT',
        'ISAAC_ROS_WS',
    ):
        environment.pop(name, None)
    environment.update(
        {
            'PATH': f'{bin_dir}{os.pathsep}{environment["PATH"]}',
            'ROS_DISTRO': 'capture-test-no-ros',
            'ISAAC_ROS_WS': str(workspace),
            'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT': str(assets_root),
            'CAPTURE_OUTPUT_ROOT': str(output_root),
            'CAPTURE_PLAYBACK_RATE': '0.25',
            'CAPTURE_DRAIN_SECONDS': '0',
            'CAPTURE_MIN_MESSAGES': '1',
            'CAPTURE_STOP_GRACE_SECONDS': str(stop_grace_seconds),
            'CAPTURE_STOP_TERM_SECONDS': str(stop_term_seconds),
            'CAPTURE_FIXTURE_MODE': mode,
            'CAPTURE_FIXTURE_RECORD_MODE': record_mode,
            'CAPTURE_FIXTURE_PLAY_STATUS': str(bag_play_status),
        }
    )
    environment.update({name: str(path) for name, path in fixture_paths.items()})
    return environment, output_root, fixture_paths


def _kill_nvidia_fixture_processes(fixture_paths):
    process_groups = set()
    for name, path in fixture_paths.items():
        if not name.endswith('_PID_FILE'):
            continue
        try:
            pid = int(path.read_text())
            process_group = os.getpgid(pid)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        if process_group != os.getpgrp():
            process_groups.add(process_group)
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_nvidia_capture(
    root,
    *,
    mode='clean',
    bag_play_status=0,
    record_mode='clean',
    stop_grace_seconds=1,
    stop_term_seconds=1,
    profiled=False,
):
    environment, output_root, fixture_paths = _create_nvidia_capture_fixture(
        root,
        mode,
        bag_play_status,
        record_mode,
        stop_grace_seconds,
        stop_term_seconds,
    )
    if profiled:
        environment['CAPTURE_NSYS_OUTPUT'] = str(root / 'profile')
    command = [str(SCRIPT_PATH), 'rtdetr-managed', f'fixture-{mode}']
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired as timeout:
        _kill_nvidia_fixture_processes(fixture_paths)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise AssertionError(f'NVIDIA capture fixture timed out:\n{stdout}\n{stderr}') from timeout
    finally:
        _kill_nvidia_fixture_processes(fixture_paths)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        output_root,
        fixture_paths,
    )


def test_nvidia_capture_rejects_component_crash_hidden_by_zero_wrapper(tmp_path):
    result, output_root, fixture_paths = _run_nvidia_capture(tmp_path, mode='crash')
    launch_log = output_root / 'logs' / 'fixture-crash.launch.log'

    assert result.returncode != 0, f'{result.stdout}\n{result.stderr}'
    assert 'PASS:' not in result.stdout
    assert fixture_paths['CAPTURE_FIXTURE_LAUNCH_STATUS_FILE'].read_text().strip() == '0'
    assert '[ERROR] [component_container_mt-2]: process has died ' in launch_log.read_text()
    assert 'exit code 7' in result.stdout + result.stderr


def test_nvidia_capture_accepts_clean_component_and_container_stop(tmp_path):
    result, output_root, fixture_paths = _run_nvidia_capture(tmp_path, mode='clean')
    launch_log = output_root / 'logs' / 'fixture-clean.launch.log'
    launch_text = launch_log.read_text()

    assert result.returncode == 0, f'{result.stdout}\n{result.stderr}'
    assert 'PASS:' in result.stdout
    assert fixture_paths['CAPTURE_FIXTURE_LAUNCH_STATUS_FILE'].read_text().strip() == '0'
    assert '[INFO] [component_container_mt-2]: process has finished cleanly' in launch_text
    assert '[INFO] [component_container-1]: process has finished cleanly' in launch_text
    assert (output_root / 'fixture-clean').is_dir()


@pytest.mark.parametrize('mode', ('missing', 'contradictory'))
def test_nvidia_capture_rejects_unconfirmed_or_contradictory_component_exit(tmp_path, mode):
    result, output_root, _ = _run_nvidia_capture(tmp_path, mode=mode)
    launch_text = (output_root / 'logs' / f'fixture-{mode}.launch.log').read_text()
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0, diagnostic
    assert 'PASS:' not in result.stdout
    assert 'component_container_mt-2' in launch_text
    if mode == 'missing':
        assert 'unconfirmed' in diagnostic.lower() or 'clean exit' in diagnostic.lower()
    else:
        assert 'process has finished cleanly [pid ' in launch_text
        assert 'contradict' in diagnostic.lower() or 'pid' in diagnostic.lower()


@pytest.mark.parametrize(
    ('mode', 'escalation'),
    (('term', 'SIGTERM'), ('kill', 'SIGKILL')),
)
def test_nvidia_capture_fails_when_graph_stop_requires_escalation(tmp_path, mode, escalation):
    result, _output_root, _fixture_paths = _run_nvidia_capture(tmp_path, mode=mode)
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0, diagnostic
    assert 'PASS:' not in result.stdout
    assert escalation in diagnostic


@pytest.mark.parametrize(
    ('mode', 'stop_grace_seconds', 'stop_term_seconds', 'diagnostic_fragment'),
    (
        ('term', 0, 1, 'graph required SIGTERM escalation'),
        ('kill', 0, 0, 'graph required SIGKILL escalation'),
    ),
)
def test_nvidia_capture_gates_escalation_with_zero_stop_deadline(
    tmp_path, mode, stop_grace_seconds, stop_term_seconds, diagnostic_fragment
):
    result, _output_root, _fixture_paths = _run_nvidia_capture(
        tmp_path,
        mode=mode,
        stop_grace_seconds=stop_grace_seconds,
        stop_term_seconds=stop_term_seconds,
    )
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0, diagnostic
    assert 'PASS:' not in result.stdout
    assert diagnostic_fragment in diagnostic


@pytest.mark.parametrize(
    ('record_mode', 'escalation'),
    (('term', 'SIGTERM'), ('kill', 'SIGKILL')),
)
def test_nvidia_capture_fails_when_recorder_stop_requires_escalation(
    tmp_path, record_mode, escalation
):
    result, _output_root, _fixture_paths = _run_nvidia_capture(
        tmp_path, mode='clean', record_mode=record_mode
    )
    diagnostic = result.stdout + result.stderr

    assert result.returncode != 0, diagnostic
    assert 'PASS:' not in result.stdout
    assert 'recorder required SIGTERM/SIGKILL escalation' in diagnostic
    assert escalation in diagnostic


def test_nvidia_capture_recorder_receives_int_without_installing_a_handler(tmp_path):
    result, _output_root, _fixture_paths = _run_nvidia_capture(tmp_path, record_mode='inherited')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS:' in result.stdout
    assert 'sending SIGTERM' not in result.stdout


@pytest.mark.parametrize('profiled', (False, True))
def test_nvidia_capture_signals_launch_without_double_signalling_components(tmp_path, profiled):
    result, _output_root, fixture_paths = _run_nvidia_capture(
        tmp_path, mode='relay_once', stop_grace_seconds=3, profiled=profiled
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert fixture_paths['CAPTURE_FIXTURE_COMPONENT_SIGNAL_FILE'].read_text().strip() == '1'


def test_nvidia_capture_preserves_playback_failure_during_cleanup(tmp_path):
    result, output_root, _fixture_paths = _run_nvidia_capture(
        tmp_path, mode='clean', bag_play_status=23
    )

    assert result.returncode == 23, f'{result.stdout}\n{result.stderr}'
    assert 'PASS:' not in result.stdout
    assert 'Capture failed. Logs:' in result.stdout
    assert (output_root / 'logs' / 'fixture-clean.launch.log').is_file()
    assert (output_root / 'logs' / 'fixture-clean.record.log').is_file()
    assert (output_root / 'fixture-clean').is_dir()


def test_capture_runner_rejects_an_unknown_lane_before_starting_ros():
    result = subprocess.run(
        [str(SCRIPT_PATH), 'unknown', 'capture_name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'unsupported lane' in result.stderr
    assert 'unknown' in result.stderr

@pytest.mark.parametrize('legacy_lane', ('rtdetr-c', 'yolov8-c'))
def test_capture_runner_rejects_legacy_native_lane_tokens(legacy_lane):
    result = subprocess.run(
        [str(SCRIPT_PATH), legacy_lane, 'capture_name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'unsupported lane' in result.stderr
    assert legacy_lane in result.stderr




def test_amd_capture_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'output-name' in result.stderr


def test_amd_yolov8_capture_fails_before_ros_when_asset_is_missing(tmp_path):
    environment = os.environ.copy()
    environment['OVG_ASSETS_ROOT'] = str(tmp_path / 'assets')
    environment['CAPTURE_INPUT_BAG'] = str(tmp_path / 'missing-bag')
    model_path = tmp_path / 'assets' / 'models' / 'yolov8' / 'yolov8s.onnx'
    environment['CAPTURE_MODEL_PATH'] = str(model_path)
    result = subprocess.run(
        [str(AMD_SCRIPT_PATH), 'yolov8', 'missing-model'],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert 'missing' in result.stderr
    assert str(model_path) in result.stderr


def test_transport_audit_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AUDIT_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'audit-name' in result.stderr


@pytest.mark.parametrize('model', ('yolov8', 'rtdetr'))
def test_amd_matrix_records_fixed_assets_and_relative_dataset_hashes(tmp_path, model):
    workspace = _create_matrix_workspace(tmp_path / 'workspace')
    bin_dir = _install_fake_launch_test(tmp_path)
    assets_one = tmp_path / 'assets-one'
    model_one, dataset_one = _create_assets(assets_one, model)
    assets_two = tmp_path / 'assets-two'
    model_two, dataset_two = _create_assets(assets_two, model)

    assets_one_alias = tmp_path / 'assets-one-alias'
    assets_one_alias.symlink_to(assets_one, target_is_directory=True)
    result_one, output_one, log_one = _run_matrix(
        tmp_path,
        workspace,
        bin_dir,
        model,
        f'{model}-assets-one',
        {
            'OVG_ASSETS_ROOT': str(assets_one_alias),
            'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT': str(assets_one),
        },
    )
    manifest_one = _assert_matrix_invocation(
        result_one, output_one, log_one, model, assets_one_alias
    )
    expected_dataset_hash = _expected_relative_tree_sha256(dataset_one)
    assert manifest_one['dataset_tree_sha256'] == expected_dataset_hash

    result_two, output_two, log_two = _run_matrix(
        tmp_path,
        workspace,
        bin_dir,
        model,
        f'{model}-assets-two',
        {'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT': str(assets_two)},
    )
    manifest_two = _assert_matrix_invocation(result_two, output_two, log_two, model, assets_two)
    assert manifest_two['model_sha256'] == hashlib.sha256(model_two.read_bytes()).hexdigest()
    assert manifest_two['dataset_tree_sha256'] == expected_dataset_hash
    assert model_one.read_bytes() == model_two.read_bytes()

    model_two.write_bytes(b'mutated fixture model\n')
    (dataset_two / 'sample.mcap').write_bytes(b'mutated fixture dataset\n')
    result_three, output_three, log_three = _run_matrix(
        tmp_path,
        workspace,
        bin_dir,
        model,
        f'{model}-assets-two-mutated',
        {'OVG_ASSETS_ROOT': str(assets_two)},
    )
    manifest_three = _assert_matrix_invocation(
        result_three, output_three, log_three, model, assets_two
    )
    assert manifest_three['model_sha256'] == hashlib.sha256(model_two.read_bytes()).hexdigest()
    assert manifest_three['model_sha256'] != manifest_two['model_sha256']
    assert manifest_three['dataset_tree_sha256'] == (_expected_relative_tree_sha256(dataset_two))
    assert manifest_three['dataset_tree_sha256'] != manifest_two['dataset_tree_sha256']


def test_amd_matrix_rejects_input_overrides_even_when_empty(tmp_path):
    workspace = _create_matrix_workspace(tmp_path / 'workspace')
    assets = tmp_path / 'assets'
    _create_assets(assets, 'yolov8')
    bin_dir = _install_fake_launch_test(tmp_path)

    for variable in (
        'CAPTURE_MODEL_PATH',
        'CAPTURE_INPUT_BAG',
        'ROS2_BENCHMARK_OVERRIDE_INPUT_DATA_PATH',
    ):
        for index, value in enumerate(('', str(tmp_path / 'input-override'))):
            matrix_name = f'reject-{variable.lower()}-{index}'
            result, output_dir, launch_log = _run_matrix(
                tmp_path,
                workspace,
                bin_dir,
                'yolov8',
                matrix_name,
                {'OVG_ASSETS_ROOT': str(assets)},
                {variable: value},
            )
            assert result.returncode == 2
            assert variable in result.stderr
            assert not launch_log.exists()
            assert not output_dir.exists()


def test_amd_matrix_rejects_conflicting_assets_roots_before_launch(tmp_path):
    workspace = _create_matrix_workspace(tmp_path / 'workspace')
    assets_one = tmp_path / 'assets-one'
    assets_two = tmp_path / 'assets-two'
    assets_one.mkdir()
    assets_two.mkdir()
    bin_dir = _install_fake_launch_test(tmp_path)

    result, output_dir, launch_log = _run_matrix(
        tmp_path,
        workspace,
        bin_dir,
        'yolov8',
        'reject-assets-conflict',
        {
            'OVG_ASSETS_ROOT': str(assets_one),
            'ROS2_BENCHMARK_OVERRIDE_ASSETS_ROOT': str(assets_two),
        },
    )
    assert result.returncode == 1
    assert 'point to different directories' in result.stderr
    assert not launch_log.exists()
    assert not output_dir.exists()


@pytest.mark.parametrize('invalid_input', ('model', 'model-directory', 'dataset'))
def test_amd_matrix_rejects_invalid_fixed_assets_before_launch(tmp_path, invalid_input):
    workspace = _create_matrix_workspace(tmp_path / 'workspace')
    assets = tmp_path / 'assets'
    model_path, dataset_path = _create_assets(assets, 'yolov8')
    if invalid_input == 'model':
        model_path.unlink()
        expected_error = 'required model file'
    elif invalid_input == 'model-directory':
        model_path.unlink()
        model_path.mkdir()
        (model_path / 'not-an-onnx-file').write_bytes(b'not a model\n')
        expected_error = 'required model file'
    else:
        (dataset_path / 'sample.mcap').unlink()
        dataset_path.rmdir()
        expected_error = 'required dataset directory'
    bin_dir = _install_fake_launch_test(tmp_path)

    result, output_dir, launch_log = _run_matrix(
        tmp_path,
        workspace,
        bin_dir,
        'yolov8',
        f'reject-invalid-{invalid_input}',
        {'OVG_ASSETS_ROOT': str(assets)},
    )
    assert result.returncode == 1
    assert expected_error in result.stderr
    assert not launch_log.exists()
    assert not output_dir.exists()


def test_amd_matrix_records_normal_and_linked_worktree_fingerprints(tmp_path):
    workspace = _create_matrix_workspace(tmp_path / 'workspace')
    linked_workspace = tmp_path / 'linked-worktree'
    _git(workspace, 'worktree', 'add', '-b', 'linked-fixture', str(linked_workspace))
    assert (workspace / '.git').is_dir()
    assert (linked_workspace / '.git').is_file()

    assets = tmp_path / 'assets'
    _create_assets(assets, 'yolov8')
    bin_dir = _install_fake_launch_test(tmp_path)
    for name, checkout in (
        ('normal', workspace),
        ('linked', linked_workspace),
    ):
        (checkout / 'tracked.txt').write_text('working-tree change\n')
        (checkout / 'untracked file.txt').write_text(f'{name} untracked\n')
        result, output_dir, launch_log = _run_matrix(
            tmp_path,
            checkout,
            bin_dir,
            'yolov8',
            f'worktree-{name}',
            {'OVG_ASSETS_ROOT': str(assets)},
        )
        manifest = _assert_matrix_invocation(result, output_dir, launch_log, 'yolov8', assets)

        revision = _git(checkout, 'rev-parse', 'HEAD').strip()
        diff = subprocess.run(
            ['git', '-C', str(checkout), 'diff', 'HEAD', '--binary'],
            check=True,
            capture_output=True,
        ).stdout
        untracked_hash_line = subprocess.run(
            ['sha256sum', '--', 'untracked file.txt'],
            check=True,
            capture_output=True,
            cwd=checkout,
        ).stdout
        expected_untracked_hash = hashlib.sha256(
            b'untracked file.txt\0' + untracked_hash_line
        ).hexdigest()
        assert manifest['monorepo_revision'] == revision
        assert manifest['monorepo_diff_head_binary_sha256'] == hashlib.sha256(diff).hexdigest()
        assert manifest['monorepo_untracked_paths'] == 'untracked\\ file.txt'
        assert manifest['monorepo_untracked_content_sha256'] == expected_untracked_hash
        assert manifest['monorepo_dirty'] == 'true'
