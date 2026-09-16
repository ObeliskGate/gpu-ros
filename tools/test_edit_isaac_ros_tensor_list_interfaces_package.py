#!/usr/bin/env python3
"""Synthetic-fixture tests for the semantic TensorBundle package editor."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).with_name("edit_isaac_ros_tensor_list_interfaces_package.py")


class PackageEditorTest(unittest.TestCase):
    def make_checkout(self, package_xml: str) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        package_path = root / "isaac_ros_tensor_list_interfaces" / "package.xml"
        package_path.parent.mkdir()
        package_path.write_text(package_xml, encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "fixture"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return temporary, package_path, commit

    def run_editor(self, root: Path, commit: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--upstream-root",
                str(root),
                "--expected-commit",
                commit,
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    def test_removes_only_the_selected_dependency_and_is_idempotent(self) -> None:
        fixture = """<?xml version=\"1.0\"?>\n<package format=\"3\">\n  <name>isaac_ros_tensor_list_interfaces</name>\n  <version>0.1.0</version>\n  <description>fixture</description>\n  <build_depend>ament_cmake_auto</build_depend>\n  <build_depend>isaac_ros_common</build_depend>\n  <test_depend>ament_lint_auto</test_depend>\n</package>\n"""
        temporary, package_path, commit = self.make_checkout(fixture)
        self.addCleanup(temporary.cleanup)

        first = self.run_editor(Path(temporary.name), commit)
        self.assertEqual(first.returncode, 0, first.stderr)
        document = ET.parse(package_path)
        self.assertEqual(
            [element.text for element in document.findall(".//build_depend")],
            ["ament_cmake_auto"],
        )
        before_second_run = package_path.read_bytes()
        second = self.run_editor(Path(temporary.name), commit)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(package_path.read_bytes(), before_second_run)

    def test_rejects_wrong_package_and_ambiguous_dependency(self) -> None:
        fixture = """<package format=\"3\"><name>wrong</name><version>1</version><build_depend>isaac_ros_common</build_depend></package>"""
        temporary, _, commit = self.make_checkout(fixture)
        self.addCleanup(temporary.cleanup)
        result = self.run_editor(Path(temporary.name), commit)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected package name", result.stderr)

        fixture_path = Path(temporary.name) / "isaac_ros_tensor_list_interfaces" / "package.xml"
        fixture_path.write_text(
            """<package format=\"3\"><name>isaac_ros_tensor_list_interfaces</name><version>1</version><build_depend>isaac_ros_common</build_depend><build_depend>isaac_ros_common</build_depend></package>""",
            encoding="utf-8",
        )
        ambiguous = self.run_editor(Path(temporary.name), commit)
        self.assertNotEqual(ambiguous.returncode, 0)
        self.assertIn("multiple", ambiguous.stderr)

    def test_rejects_wrong_upstream_revision(self) -> None:
        fixture = """<package format=\"3\"><name>isaac_ros_tensor_list_interfaces</name><version>1</version></package>"""
        temporary, _, _ = self.make_checkout(fixture)
        self.addCleanup(temporary.cleanup)
        result = self.run_editor(Path(temporary.name), "0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not the expected revision", result.stderr)


if __name__ == "__main__":
    unittest.main()
