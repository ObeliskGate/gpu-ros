#!/usr/bin/env python3
"""Remove only Isaac ROS version-stamp build dependencies from a pinned checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VERSION_STAMP_BLOCK = (
    "# Embed versioning information into installed files\n"
    "ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)\n"
    'include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")\n'
    "generate_version_info(${PROJECT_NAME})\n"
    "\n"
)
ISAAC_ROS_COMMON_BUILD_DEPENDENCY = "  <build_depend>isaac_ros_common</build_depend>\n"


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    contents = path.read_text(encoding="utf-8")
    count = contents.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one {description}, found {count}; "
            "the pinned source does not match the reviewed standalone patch contract."
        )
    path.write_text(contents.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    if len(args.expected_commit) != 40 or any(
        char not in "0123456789abcdef" for char in args.expected_commit
    ):
        parser.error("--expected-commit must be a full lowercase Git SHA-1")

    source = args.source_root.resolve(strict=True)
    actual_commit = git(source, "rev-parse", "--verify", "HEAD")
    if actual_commit != args.expected_commit:
        raise RuntimeError(
            f"source HEAD {actual_commit} does not match the image-paired commit "
            f"{args.expected_commit}"
        )
    if git(source, "status", "--porcelain"):
        raise RuntimeError(f"source checkout is not pristine: {source}")

    for relative in (
        "ros2_benchmark/CMakeLists.txt",
        "ros2_benchmark_interfaces/CMakeLists.txt",
    ):
        replace_once(
            source / relative,
            VERSION_STAMP_BLOCK,
            "",
            "Isaac ROS version-stamp CMake block",
        )

    for relative in (
        "ros2_benchmark/package.xml",
        "ros2_benchmark_interfaces/package.xml",
    ):
        replace_once(
            source / relative,
            ISAAC_ROS_COMMON_BUILD_DEPENDENCY,
            "",
            "isaac_ros_common build dependency",
        )

    subprocess.run(["git", "-C", str(source), "diff", "--check"], check=True)
    changed = set(git(source, "diff", "--name-only").splitlines())
    expected = {
        "ros2_benchmark/CMakeLists.txt",
        "ros2_benchmark/package.xml",
        "ros2_benchmark_interfaces/CMakeLists.txt",
        "ros2_benchmark_interfaces/package.xml",
    }
    if changed != expected:
        raise RuntimeError(
            f"standalone adaptation changed unexpected files: {sorted(changed ^ expected)}"
        )

    print(f"Prepared standalone ros2_benchmark source at {actual_commit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
