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

from pathlib import Path
import subprocess


SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_fixed_input_capture.sh'
)
AUDIT_SCRIPT_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'run_nvidia_yolov8_transport_audit.sh'
)


def test_capture_runner_is_executable_and_has_valid_bash_syntax():
    assert SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(SCRIPT_PATH)], check=True)


def test_capture_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<rtdetr-c|rtdetr-managed|yolov8-c|yolov8-managed>' in result.stdout


def test_capture_runner_rejects_an_unknown_lane_before_starting_ros():
    result = subprocess.run(
        [str(SCRIPT_PATH), 'unknown', 'capture_name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsupported lane 'unknown'" in result.stderr


def test_transport_audit_runner_is_executable_and_has_valid_bash_syntax():
    assert AUDIT_SCRIPT_PATH.stat().st_mode & 0o111
    subprocess.run(['bash', '-n', str(AUDIT_SCRIPT_PATH)], check=True)
    assert AUDIT_SCRIPT_PATH.read_text().count('nsys stats \\\n  --quiet') == 2
    assert AUDIT_SCRIPT_PATH.read_text().count('--format json:mem=B') == 2


def test_transport_audit_runner_help_does_not_require_ros_environment():
    result = subprocess.run(
        [str(AUDIT_SCRIPT_PATH), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '<audit-name>' in result.stdout


def test_transport_audit_runner_rejects_an_invalid_name_before_starting_ros():
    result = subprocess.run(
        [str(AUDIT_SCRIPT_PATH), 'invalid/name'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'audit-name may contain only' in result.stderr
