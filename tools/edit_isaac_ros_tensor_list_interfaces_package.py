#!/usr/bin/env python3
"""Apply the vendor-neutral TensorBundle package metadata change semantically.

The script operates on a user-provided checkout of the exact upstream
``isaac_ros_common`` revision.  It deliberately contains no upstream
``package.xml`` text or license block: only the selected dependency element is
removed from the user's local XML tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


EXPECTED_UPSTREAM_COMMIT = "6ff6aec39a87179d7bb5b986c8aad1b52e56dc53"
DEFAULT_PACKAGE_NAME = "isaac_ros_tensor_list_interfaces"
DEFAULT_PACKAGE_XML = Path("isaac_ros_tensor_list_interfaces/package.xml")
DEPENDENCY_TAGS = {
    "build_depend",
    "build_export_depend",
    "depend",
    "exec_depend",
    "test_depend",
}


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def upstream_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot verify upstream Git revision in {root}: {exc}")
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail(f"upstream revision is not a full commit id: {commit!r}")
    return commit


def parse_package(path: Path) -> ET.ElementTree:
    if not path.is_file():
        fail(f"package XML does not exist: {path}")
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        return ET.parse(path, parser=parser)
    except ET.ParseError as exc:
        fail(f"invalid package XML {path}: {exc}")


def text_of(element: ET.Element) -> str:
    return (element.text or "").strip()


def find_dependency_elements(root: ET.Element, name: str) -> list[tuple[ET.Element | None, ET.Element]]:
    matches: list[tuple[ET.Element | None, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag in DEPENDENCY_TAGS and text_of(child) == name:
                matches.append((parent, child))
    return matches


def validate_and_edit(
    package_xml: Path,
    package_name: str,
) -> bool:
    tree = parse_package(package_xml)
    root = tree.getroot()
    if root.tag != "package":
        fail(f"unexpected XML root {root.tag!r}; expected 'package'")

    names = [element for element in root.findall("name")]
    if len(names) != 1 or text_of(names[0]) != package_name:
        actual = text_of(names[0]) if len(names) == 1 else "<ambiguous>"
        fail(f"expected package name {package_name!r}, found {actual!r}")

    versions = [element for element in root.findall("version")]
    if len(versions) != 1 or not text_of(versions[0]):
        fail("package XML must contain exactly one non-empty version element")

    matches = find_dependency_elements(root, "isaac_ros_common")
    unexpected = [element.tag for _, element in matches if element.tag != "build_depend"]
    if unexpected:
        fail(
            "isaac_ros_common appears in unexpected dependency element(s): "
            + ", ".join(unexpected)
        )
    if len(matches) > 1:
        fail("ambiguous package XML: multiple isaac_ros_common dependencies")
    if not matches:
        return False

    parent, element = matches[0]
    if parent is None:
        fail("dependency element has no XML parent")
    parent.remove(element)
    ET.indent(tree, space="  ")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{package_xml.name}.", dir=package_xml.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
        os.replace(temporary_path, package_xml)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        required=True,
        type=Path,
        help="root of the user-provided isaac_ros_common checkout",
    )
    parser.add_argument(
        "--expected-commit",
        default=EXPECTED_UPSTREAM_COMMIT,
        help="full upstream commit expected at HEAD",
    )
    parser.add_argument(
        "--package-name",
        default=DEFAULT_PACKAGE_NAME,
        help="package name to validate",
    )
    parser.add_argument(
        "--package-xml",
        type=Path,
        help="optional package XML path, relative to --upstream-root unless absolute",
    )
    args = parser.parse_args(argv)

    root = args.upstream_root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        fail("--expected-commit must be a full 40-character lowercase commit id")
    actual_commit = upstream_commit(root)
    if actual_commit != args.expected_commit:
        fail(
            f"upstream checkout is not the expected revision: "
            f"expected {args.expected_commit}, found {actual_commit}"
        )

    package_xml = args.package_xml or DEFAULT_PACKAGE_XML
    if not package_xml.is_absolute():
        package_xml = root / package_xml
    changed = validate_and_edit(package_xml.resolve(), args.package_name)
    print(
        f"{'removed' if changed else 'already absent'} "
        f"isaac_ros_common build dependency from {package_xml}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
