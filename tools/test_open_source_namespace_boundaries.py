#!/usr/bin/env python3
"""Static checks for project naming and the NVIDIA compatibility boundary.

This script intentionally uses only the Python standard library so it can run
before a ROS installation or colcon workspace exists.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOTS = tuple(
    ROOT / name
    for name in (
        "gpu_ros_managed",
        "gpu_ros_object_detection",
    )
)
BENCHMARKS_ROOT = ROOT / "gpu_ros_object_detection" / "benchmarks"
MANAGED_TRANSPORT_MANIFEST = (
    ROOT / "gpu_ros_managed" / "gpu_ros_managed_tensor_bundle" / "package.xml"
)
BUFFER_ACCESS_MANIFEST = (
    ROOT / "gpu_ros_managed" / "gpu_ros_rosidl_buffer" / "package.xml"
)
COMPAT_PACKAGE = "gpu_ros_nvidia_tensor_bundle_compat"
REFERENCE_PACKAGE = "gpu_ros_nvidia_reference"
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
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".hip",
    ".py",
    ".sh",
    ".xml",
    ".msg",
    ".srv",
    ".md",
    ".yaml",
    ".yml",
}


def text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and (path.name == "package.xml" or path.suffix in TEXT_SUFFIXES):
            yield path


def project_text_files():
    for project_root in PROJECT_ROOTS:
        yield from text_files(project_root)


def project_package_files() -> list[Path]:
    return sorted(
        package_xml
        for project_root in PROJECT_ROOTS
        for package_xml in project_root.rglob("package.xml")
    )


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


def assert_project_roots() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in PROJECT_ROOTS if not path.is_dir()]
    assert not missing, "required project roots are missing: " + ", ".join(missing)

    benchmark_packages = sorted(BENCHMARKS_ROOT.rglob("package.xml"))
    assert not benchmark_packages, (
        "gpu_ros_object_detection/benchmarks must remain a non-package directory:\n"
        + "\n".join(str(path) for path in benchmark_packages)
    )


def assert_package_names() -> None:
    package_files = project_package_files()
    assert package_files, "no project package manifests found"
    found_legacy = []
    tensor_list_dependents = []
    core_model_isaac_dependents = []
    for package_xml in package_files:
        name = package_name(package_xml)
        dependencies = package_dependencies(package_xml)
        if name in LEGACY_PACKAGES or name.startswith("isaac_ros_"):
            found_legacy.append(f"{package_xml}: {name}")
        if name != package_xml.parent.name:
            raise AssertionError(f"package directory/name mismatch: {package_xml}: {name}")
        if "isaac_ros_tensor_msgs" in dependencies:
            tensor_list_dependents.append(name)
        if name in {"gpu_ros_rtdetr", "gpu_ros_yolov8"}:
            isaac_dependencies = sorted(
                dependency for dependency in dependencies if dependency.startswith("isaac_ros_")
            )
            if isaac_dependencies:
                core_model_isaac_dependents.append(
                    f"{package_xml}: {', '.join(isaac_dependencies)}"
                )
    assert not found_legacy, "project-owned legacy package names remain:\n" + "\n".join(
        found_legacy
    )
    assert tensor_list_dependents == [COMPAT_PACKAGE], (
        "only the NVIDIA compatibility package may depend on "
        f"isaac_ros_tensor_msgs; found {tensor_list_dependents}"
    )
    assert not core_model_isaac_dependents, (
        "core model packages must not have non-test Isaac ROS dependencies:\n"
        + "\n".join(core_model_isaac_dependents)
    )


def assert_local_transport_manifest() -> None:
    assert MANAGED_TRANSPORT_MANIFEST.is_file(), (
        f"required local managed transport manifest is missing: {MANAGED_TRANSPORT_MANIFEST}"
    )
    assert package_name(MANAGED_TRANSPORT_MANIFEST) == MANAGED_TRANSPORT_MANIFEST.parent.name, (
        f"managed transport manifest/package directory mismatch: {MANAGED_TRANSPORT_MANIFEST}"
    )
    assert not (ROOT / "gpu_ros_managed" / "gpu_ros_managed_tensor_list").exists(), (
        "old managed TensorList package directory remains"
    )
    assert BUFFER_ACCESS_MANIFEST.is_file(), (
        f"required rosidl Buffer access manifest is missing: {BUFFER_ACCESS_MANIFEST}"
    )
    assert package_name(BUFFER_ACCESS_MANIFEST) == BUFFER_ACCESS_MANIFEST.parent.name, (
        "rosidl Buffer access manifest/package directory mismatch: "
        f"{BUFFER_ACCESS_MANIFEST}"
    )


def assert_legacy_namespaces_absent() -> None:
    failures = []
    for path in project_text_files():
        content = path.read_text(encoding="utf-8")
        for legacy in LEGACY_NAMESPACES:
            if legacy in content:
                failures.append(f"{path}: {legacy}")
    assert not failures, "project-owned legacy C++ namespaces remain:\n" + "\n".join(failures)


def assert_nvidia_namespace_allowlist() -> None:
    allowed_external_prefixes = (
        "nvidia::isaac_ros::dnn_inference::",
        "nvidia::isaac_ros::image_proc::",
        "nvidia::isaac_ros::rtdetr::",
        "nvidia::isaac_ros::yolov8::",
    )
    failures = []
    for path in project_text_files():
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
        ROOT / "gpu_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml",
        ROOT / "gpu_ros_object_detection/docker/phase2a-amd.Dockerfile",
        ROOT / "gpu_ros_object_detection/docker/phase2a-amd-entrypoint.sh",
        ROOT / "gpu_ros_object_detection/tools/phase2",
    )
    for path in amd_files:
        content = path.read_text(encoding="utf-8")
        assert COMPAT_PACKAGE not in content, f"AMD profile resolves compat package: {path}"
        assert REFERENCE_PACKAGE not in content, f"AMD profile resolves reference package: {path}"


def assert_rosidl_buffer_profiles() -> None:
    profiles = {
        ROOT / "gpu_ros_object_detection/docker/colcon-defaults-phase2b-nvidia.yaml": (
            "src/gpu-ros/gpu_ros_managed/gpu_ros_rosidl_buffer",
            "-DGPU_ROS_ROSIDL_BUFFER_BUILD_CUDA=ON",
        ),
        ROOT / "gpu_ros_object_detection/docker/colcon-defaults-phase2a-amd.yaml": (
            "gpu_ros_managed/gpu_ros_rosidl_buffer",
            "-DGPU_ROS_ROSIDL_BUFFER_BUILD_CUDA=OFF",
        ),
        ROOT / ".github/workflows/cpu-tests.yml": (
            "gpu_ros_managed/gpu_ros_rosidl_buffer",
            "-DGPU_ROS_ROSIDL_BUFFER_BUILD_CUDA=OFF",
        ),
    }
    for path, (package_path, cuda_option) in profiles.items():
        content = path.read_text(encoding="utf-8")
        assert package_path in content, f"Buffer access package missing from profile: {path}"
        assert cuda_option in content, f"Buffer CUDA selection missing from profile: {path}"


def assert_no_old_project_directories() -> None:
    for project_root in PROJECT_ROOTS:
        for legacy in LEGACY_PACKAGES:
            assert not (project_root / legacy).exists(), (
                f"old project directory remains: {project_root / legacy}"
            )


def main() -> int:
    assert_project_roots()
    assert_package_names()
    assert_local_transport_manifest()
    assert_legacy_namespaces_absent()
    assert_nvidia_namespace_allowlist()
    assert_amd_profile_excludes_compat()
    assert_rosidl_buffer_profiles()
    assert_no_old_project_directories()
    print("open-source namespace and dependency boundary checks: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ElementTree.ParseError) as error:
        print(
            f"open-source namespace and dependency boundary checks: FAIL: {error}", file=sys.stderr
        )
        raise SystemExit(1)
