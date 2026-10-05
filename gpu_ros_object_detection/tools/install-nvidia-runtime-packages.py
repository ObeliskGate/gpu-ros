#!/usr/bin/env python3
"""Add the image-paired graph dependencies without changing any installed package."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def run(*args):
    return subprocess.check_output(args, text=True, env={**os.environ, "LC_ALL": "C"})


def inventory():
    entries = run("dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n")
    return {
        fields[0].split(":", 1)[0]: fields[1]
        for line in entries.splitlines()
        if len(fields := line.split("\t")) == 3 and fields[2] == "installed"
    }


def main():
    lock_path = Path(sys.argv[1])
    lock = {}
    for line in lock_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version, digest = line.split()
        if name in lock:
            raise RuntimeError(f"Duplicate package pin: {name}")
        lock[name] = (version, digest)

    if run("dpkg", "--print-architecture").strip() != "amd64":
        raise RuntimeError("The NVIDIA runtime lock is for amd64 only")
    before = inventory()
    evidence = Path("/var/lib/gpu-ros")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "nvidia-packages-before.json").write_text(json.dumps(before, indent=2) + "\n")
    roots = [
        "libcudnn9-cuda-13", "ros-lyrical-isaac-ros-benchmark",
        "ros-lyrical-ros2-benchmark", "ros-lyrical-isaac-ros-rtdetr",
        "ros-lyrical-isaac-ros-yolov8", "ros-lyrical-isaac-ros-dnn-image-encoder",
        "ros-lyrical-isaac-ros-image-proc", "ros-lyrical-isaac-ros-tensor-proc",
        "ros-lyrical-isaac-ros-tensor-rt", "ros-lyrical-cuda-buffer",
        "ros-lyrical-cuda-buffer-backend-msgs", "ros-lyrical-isaac-ros-tensor-msgs",
        "ros-lyrical-tensor-msgs",
        "ros-lyrical-isaac-ros-test",
    ]
    # Prefer every already-installed version, including non-ROS libraries. Exact
    # addition pins prevent a future release catalog from selecting new ABIs.
    preferences = []
    for name, version in sorted(before.items()):
        preferences.append(f"Package: {name}\nPin: version {version}\nPin-Priority: 1001\n")
    for name, (version, _) in sorted(lock.items()):
        if name not in before:
            preferences.append(f"Package: {name}\nPin: version {version}\nPin-Priority: 1001\n")
        elif before[name] != version:
            raise RuntimeError(f"Baseline differs from the paired closure: {name}={before[name]}, expected {version}")
    preference_path = Path("/etc/apt/preferences.d/gpu-ros-nvidia-runtime")
    preference_path.write_text("\n".join(preferences))
    specs = [f"{name}={lock[name][0]}" for name in roots]
    apt = ["apt-get", "--no-install-recommends", "--no-remove"]
    plan = run(*apt, "--simulate", "install", *specs)
    print(plan, end="", flush=True)
    (evidence / "nvidia-packages-plan.txt").write_text(plan)
    additions = {}
    for line in plan.splitlines():
        if line.startswith("Remv "):
            raise RuntimeError(f"Refusing package removal: {line}")
        if not line.startswith("Inst "):
            continue
        match = re.match(r"Inst (\S+) (?:\[[^]]+\] )?\((\S+)", line)
        if not match:
            raise RuntimeError(f"Unrecognized APT transaction: {line}")
        name, version = match.groups()
        name = name.split(":", 1)[0]
        if name in before:
            raise RuntimeError(f"Refusing to replace an installed package: {line}")
        if name not in lock or lock[name][0] != version:
            raise RuntimeError(f"Unpinned dependency: {line}")
        if name == "ros-lyrical-cuda-buffer-backend" or re.match(
            r"(?:nvidia-(?:driver|dkms|kernel)|cuda-drivers|libcuda1|nvidia-cuda-)", name
        ) or "cuda-12" in name or name.endswith("-12-0"):
            raise RuntimeError(f"Prohibited CUDA/driver dependency: {line}")
        metadata = run("apt-cache", "show", f"{name}={version}")
        hashes = set(re.findall(r"^SHA256: ([0-9a-f]{64})$", metadata, re.MULTILINE))
        expected = lock[name][1]
        if len(hashes) != 1 or (expected != "-" and hashes != {expected}):
            raise RuntimeError(f"APT metadata hash mismatch: {name}={version}: {hashes}")
        additions[name] = (version, hashes.pop())

    # APT verifies signed repository metadata; also verify the actual archives
    # against the selected metadata and the explicit CUDA/Isaac hashes above.
    print(run(*apt, "--yes", "--download-only", "install", *specs), end="", flush=True)
    verified = set()
    for archive in Path("/var/cache/apt/archives").glob("*.deb"):
        name = run("dpkg-deb", "-f", str(archive), "Package").strip()
        if name not in additions:
            continue
        version = run("dpkg-deb", "-f", str(archive), "Version").strip()
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (version, digest) != additions[name]:
            raise RuntimeError(f"Downloaded archive mismatch: {archive}")
        verified.add(name)
    if verified != additions.keys():
        raise RuntimeError(f"Missing verified archives: {additions.keys() - verified}")
    print(run(*apt, "--yes", "--no-download", "install", *specs), end="", flush=True)
    after = inventory()
    if any(after.get(name) != version for name, version in before.items()):
        raise RuntimeError("Installation changed the baseline package inventory")
    if {n: v for n, v in after.items() if n not in before} != {
        n: v for n, (v, _) in additions.items()
    }:
        raise RuntimeError("Installed additions differ from the approved transaction")
    (evidence / "nvidia-packages-added.json").write_text(json.dumps(additions, indent=2) + "\n")
    (evidence / "nvidia-packages-after.json").write_text(json.dumps(after, indent=2) + "\n")
    # Do not permanently hold unrelated base packages in a derived development image.
    preference_path.unlink()


if __name__ == "__main__":
    main()
