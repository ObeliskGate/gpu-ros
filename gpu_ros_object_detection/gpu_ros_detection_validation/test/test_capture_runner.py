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
import subprocess
from pathlib import Path

import pytest

MATRIX_SCRIPT_PATH = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'run_amd_phase2b_benchmark_matrix.sh'
)
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
    for graph in sorted(
        {graph for graphs in MATRIX_GRAPH_NAMES.values() for graph in graphs}
    ):
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
    output_dir = (
        root / 'results' / 'phase2b-benchmark-matrix' / matrix_name
    )
    return result, output_dir, launch_log


def _read_matrix_manifest(output_dir):
    return {
        key: value
        for key, _, value in (
            line.partition('=')
            for line in (output_dir / 'matrix_manifest.txt')
            .read_text()
            .splitlines()
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
    records = [
        line.split('\t')
        for line in launch_log.read_text().splitlines()
    ]
    expected_indices = (0, 1, 2, 1, 2, 0, 2, 0, 1)
    assert len(records) == 9
    assert [record[0] for record in records] == [
        MATRIX_GRAPH_NAMES[model][index] for index in expected_indices
    ]
    assert all(len(record) == 2 for record in records)
    assert {record[1] for record in records} == {str(assets_root)}
    manifest = _read_matrix_manifest(output_dir)
    model_path = assets_root / MATRIX_MODEL_PATHS[model]
    dataset_path = (
        assets_root / 'datasets' / 'r2bdataset2024_v1' / 'r2b_robotarm'
    )
    assert manifest['schema_version'] == '3'
    assert manifest['assets_root'] == str(assets_root)
    assert manifest['model_path'] == str(model_path)
    assert manifest['model_sha256'] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert manifest['dataset_path'] == str(dataset_path)
    assert (
        manifest['dataset_tree_hash_algorithm']
        == 'sha256sum-sorted-relative-path-v1'
    )
    for result_file in output_dir.glob('round-*.json'):
        assert json.loads(result_file.read_text()) == {'fixture': True}
    assert len(list(output_dir.glob('round-*.json'))) == 9
    return manifest

SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_fixed_input_capture.sh'
AMD_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_amd_phase2a_fixed_input_capture.sh'
AUDIT_SCRIPT_PATH = Path(__file__).parents[1] / 'scripts' / 'run_nvidia_yolov8_transport_audit.sh'


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
    manifest_two = _assert_matrix_invocation(
        result_two, output_two, log_two, model, assets_two
    )
    assert manifest_two['model_sha256'] == hashlib.sha256(
        model_two.read_bytes()
    ).hexdigest()
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
    assert manifest_three['model_sha256'] == hashlib.sha256(
        model_two.read_bytes()
    ).hexdigest()
    assert manifest_three['model_sha256'] != manifest_two['model_sha256']
    assert manifest_three['dataset_tree_sha256'] == (
        _expected_relative_tree_sha256(dataset_two)
    )
    assert (
        manifest_three['dataset_tree_sha256']
        != manifest_two['dataset_tree_sha256']
    )


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
        manifest = _assert_matrix_invocation(
            result, output_dir, launch_log, 'yolov8', assets
        )

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
        assert manifest['monorepo_diff_head_binary_sha256'] == hashlib.sha256(
            diff
        ).hexdigest()
        assert manifest['monorepo_untracked_paths'] == 'untracked\\ file.txt'
        assert (
            manifest['monorepo_untracked_content_sha256']
            == expected_untracked_hash
        )
        assert manifest['monorepo_dirty'] == 'true'
