#!/usr/bin/env python3
"""Static checks for project naming and the NVIDIA compatibility boundary.

This script intentionally uses only the Python standard library so it can run
before a ROS installation or colcon workspace exists.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATED = ROOT / "migrated_packages"
COMPAT_PACKAGE = "gpu_ros_nvidia_tensor_bundle_compat"
LEGACY_PACKAGES = {
    "isaac_ros_detection_common",
    "isaac_ros_detection_validation",
    "isaac_ros_onnx_inference",
    "isaac_ros_rtdetr_std",
    "isaac_ros_yolov8_std",
    "gpu_ros_managed_tensor_list",
}
LEGACY_NAMESPACES = {
    "nvidia::isaac_ros::onnx_inference",
    "nvidia::isaac_ros::rtdetr_std",
    "nvidia::isaac_ros::yolov8_std",
    "nvidia::isaac_ros::detection_common",
}
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".hip", ".py", ".sh",
    ".xml", ".msg", ".srv", ".md", ".yaml", ".yml",
}


def text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and (path.name == "package.xml" or path.suffix in TEXT_SUFFIXES):
            yield path


def package_name(package_xml: Path) -> str:
    root = ElementTree.parse(package_xml).getroot()
    name = root.findtext("name")
    if not name:
        raise AssertionError(f"package has no <name>: {package_xml}")
    return name


def package_dependencies(package_xml: Path) -> set[str]:
    root = ElementTree.parse(package_xml).getroot()
    dependencies: set[str] = set()
    for element in root:
        if element.tag in {"depend", "build_depend", "build_export_depend", "exec_depend"}:
            if element.text:
                dependencies.add(element.text.strip())
    return dependencies


def assert_package_names() -> None:
    package_files = sorted(MIGRATED.rglob("package.xml"))
    assert package_files, "no migrated package manifests found"
    found_legacy = []
    tensor_list_dependents = []
    for package_xml in package_files:
        name = package_name(package_xml)
        if name in LEGACY_PACKAGES or name.startswith("isaac_ros_"):
            found_legacy.append(f"{package_xml}: {name}")
        if name != package_xml.parent.name:
            raise AssertionError(f"package directory/name mismatch: {package_xml}: {name}")
        if "isaac_ros_tensor_list_interfaces" in package_dependencies(package_xml):
            tensor_list_dependents.append(name)
    assert not found_legacy, "project-owned legacy package names remain:\n" + "\n".join(found_legacy)
    assert tensor_list_dependents == [COMPAT_PACKAGE], (
        "only the NVIDIA compatibility package may depend on "
        f"isaac_ros_tensor_list_interfaces; found {tensor_list_dependents}"
    )

    sibling_root = Path(
        os.environ.get("GPU_ROS_MANAGED_DIR", str(ROOT.parent / "gpu_ros_managed"))
    )
    sibling_manifest = sibling_root / "gpu_ros_managed_tensor_bundle" / "package.xml"
    if sibling_manifest.is_file():
        assert package_name(sibling_manifest) == "gpu_ros_managed_tensor_bundle"
        assert not (sibling_root / "gpu_ros_managed_tensor_list").exists(), (
            "old managed TensorList package directory remains"
        )


def assert_legacy_namespaces_absent() -> None:
    failures = []
    for path in text_files(MIGRATED):
        content = path.read_text(encoding="utf-8")
        for legacy in LEGACY_NAMESPACES:
            if legacy in content:
                failures.append(f"{path}: {legacy}")
    assert not failures, "project-owned legacy C++ namespaces remain:\n" + "\n".join(failures)


def assert_nvidia_namespace_allowlist() -> None:
    allowed_external_prefixes = (
        "nvidia::isaac_ros::dnn_inference::",
        "nvidia::isaac_ros::image_proc::",
        "nvidia::isaac_ros::nitros",
        "nvidia::isaac_ros::rtdetr::",
        "nvidia::isaac_ros::yolov8::",
    )
    failures = []
    for path in text_files(MIGRATED):
        relative = path.relative_to(ROOT).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "nvidia::isaac_ros::" not in line:
                continue
            if any(prefix in line for prefix in allowed_external_prefixes):
                continue
            failures.append(f"{relative}:{line_number}: {line.strip()}")
    assert not failures, "unallowlisted NVIDIA namespace use:\n" + "\n".join(failures)


def assert_amd_profile_excludes_compat() -> None:
    amd_files = (
        ROOT / "docker/colcon-defaults-phase2a-amd.yaml",
        ROOT / "docker/phase2a-amd.Dockerfile",
        ROOT / "docker/phase2a-amd-entrypoint.sh",
        ROOT / "tools/phase2",
    )
    for path in amd_files:
        content = path.read_text(encoding="utf-8")
        assert COMPAT_PACKAGE not in content, f"AMD profile resolves compat package: {path}"
        assert "isaac_ros_tensor_list_interfaces" not in content, (
            f"AMD profile resolves NVIDIA TensorList interface: {path}"
        )
    profile = amd_files[0].read_text(encoding="utf-8")
    assert "migrated_packages/gpu_ros_tensor_bundle_msgs" in profile


def assert_no_old_project_directories() -> None:
    for legacy in LEGACY_PACKAGES:
        assert not (MIGRATED / legacy).exists(), f"old project directory remains: {legacy}"


def main() -> int:
    assert_package_names()
    assert_legacy_namespaces_absent()
    assert_nvidia_namespace_allowlist()
    assert_amd_profile_excludes_compat()
    assert_no_old_project_directories()
    print("open-source namespace and dependency boundary checks: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ElementTree.ParseError) as error:
        print(f"open-source namespace and dependency boundary checks: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
