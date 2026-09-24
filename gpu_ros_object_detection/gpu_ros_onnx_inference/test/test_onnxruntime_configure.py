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

"""Configure the package's actual ORT discovery code without ROS dependencies."""

import os
import subprocess
from pathlib import Path

import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
HEADER = 'onnxruntime_cxx_api.h'
LIBRARY = 'libonnxruntime.so'


def fixture_project(tmp_path):
    source = (PACKAGE_DIR / 'CMakeLists.txt').read_text(encoding='utf-8')
    start = source.index('set(ONNXRUNTIME_ROOT "')
    end = source.index('option(ORT_ENABLE_CUDA', start)
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.22.1)\n'
        'project(ort_find_fixture LANGUAGES NONE)\n\n' + source[start:end],
        encoding='utf-8',
    )
    return project


def installation(path, *, header=True, library=True):
    if header:
        (path / 'include').mkdir(parents=True, exist_ok=True)
        (path / 'include' / HEADER).touch()
    if library:
        (path / 'lib').mkdir(parents=True, exist_ok=True)
        (path / 'lib' / LIBRARY).touch()
    return path


def configure(project, build, *options):
    env = os.environ.copy()
    for name in (
        'ONNXRUNTIME_ROOT',
        'ONNXRUNTIME_INCLUDE_DIR',
        'ONNXRUNTIME_LIBRARY',
        'CMAKE_PREFIX_PATH',
        'CMAKE_INCLUDE_PATH',
        'CMAKE_LIBRARY_PATH',
    ):
        env.pop(name, None)
    return subprocess.run(
        ['cmake', '-S', str(project), '-B', str(build), *options],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def cache_value(build, name):
    for line in (build / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith(name + ':'):
            return line.partition('=')[2]
    raise AssertionError(f'{name} absent from CMakeCache.txt')


def assert_selection(result, build, include, library):
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'-- ONNX Runtime include dir: {include}\n' in result.stdout
    assert f'-- ONNX Runtime library: {library}\n' in result.stdout
    assert cache_value(build, 'ONNXRUNTIME_INCLUDE_DIR') == str(include)
    assert cache_value(build, 'ONNXRUNTIME_LIBRARY') == str(library)


@pytest.mark.parametrize('empty_cache', [False, True])
def test_root_only_resolves_both_files(tmp_path, empty_cache):
    project = fixture_project(tmp_path)
    root = installation(tmp_path / 'root')
    build = tmp_path / 'build'
    options = [f'-DONNXRUNTIME_ROOT={root}']
    if empty_cache:
        options += ['-DONNXRUNTIME_INCLUDE_DIR=', '-DONNXRUNTIME_LIBRARY=']
    result = configure(project, build, *options)
    assert_selection(result, build, root / 'include', root / 'lib' / LIBRARY)

    # A second configure must reuse the resolved paths, even without -D overrides.
    again = configure(project, build)
    assert_selection(again, build, root / 'include', root / 'lib' / LIBRARY)


def test_independent_explicit_paths_survive_reconfigure(tmp_path):
    project = fixture_project(tmp_path)
    headers = installation(tmp_path / 'headers', library=False)
    libraries = installation(tmp_path / 'libraries', header=False)
    build = tmp_path / 'build'
    include = headers / 'include'
    library = libraries / 'lib' / LIBRARY
    result = configure(
        project,
        build,
        f'-DONNXRUNTIME_INCLUDE_DIR={include}',
        f'-DONNXRUNTIME_LIBRARY={library}',
    )
    assert_selection(result, build, include, library)
    assert_selection(configure(project, build), build, include, library)


def test_root_with_explicit_library_override(tmp_path):
    project = fixture_project(tmp_path)
    root = installation(tmp_path / 'root')
    other = installation(tmp_path / 'other', header=False)
    build = tmp_path / 'build'
    library = other / 'lib' / LIBRARY
    result = configure(
        project,
        build,
        f'-DONNXRUNTIME_ROOT={root}',
        f'-DONNXRUNTIME_LIBRARY={library}',
    )
    assert_selection(result, build, root / 'include', library)
    assert_selection(configure(project, build), build, root / 'include', library)


@pytest.mark.parametrize('header,library', [(False, False), (True, False), (False, True)])
def test_incomplete_root_cannot_use_other_installation(tmp_path, header, library):
    project = fixture_project(tmp_path)
    root = installation(tmp_path / 'root', header=header, library=library)
    decoy = installation(tmp_path / 'decoy')
    result = configure(
        project,
        tmp_path / 'build',
        f'-DONNXRUNTIME_ROOT={root}',
        f'-DCMAKE_PREFIX_PATH={decoy}',
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'ONNX Runtime was not found' in result.stderr
