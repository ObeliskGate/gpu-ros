#!/usr/bin/env python3
"""Compare CPU and MIGraphX execution of the same formal RT-DETRv2 ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images-npy", type=Path)
    parser.add_argument("--orig-target-sizes-npy", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    if ort.__version__ != "1.23.1":
        raise RuntimeError(
            f"provider parity requires ONNX Runtime 1.23.1, got {ort.__version__}"
        )

    if (args.images_npy is None) != (args.orig_target_sizes_npy is None):
        parser.error("provide both fixed-input .npy files or neither")
    if args.images_npy:
        images = np.load(args.images_npy).astype(np.float32, copy=False)
        target_sizes = np.load(args.orig_target_sizes_npy).astype(np.int64, copy=False)
    else:
        images = np.random.default_rng(20260824).random(
            (1, 3, 640, 640), dtype=np.float32
        )
        target_sizes = np.asarray([[480, 640]], dtype=np.int64)

    available = set(ort.get_available_providers())
    if "MIGraphXExecutionProvider" not in available:
        raise RuntimeError(
            f"MIGraphXExecutionProvider is unavailable; available={sorted(available)}"
        )
    cpu = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    migraphx = ort.InferenceSession(
        str(args.onnx),
        providers=[("MIGraphXExecutionProvider", {"device_id": str(args.device_id)})],
    )
    feeds = {"images": images, "orig_target_sizes": target_sizes}
    cpu_outputs = cpu.run(["labels", "boxes", "scores"], feeds)
    migraphx_outputs = migraphx.run(["labels", "boxes", "scores"], feeds)
    np.testing.assert_array_equal(migraphx_outputs[0], cpu_outputs[0])
    score_error = float(np.max(np.abs(migraphx_outputs[2] - cpu_outputs[2])))
    box_error = float(np.max(np.abs(migraphx_outputs[1] - cpu_outputs[1])))
    if score_error > 1e-3:
        raise AssertionError(f"score max absolute error {score_error} exceeds 1e-3")
    if box_error > 0.1:
        raise AssertionError(f"box max absolute error {box_error} exceeds 0.1 px")
    print(json.dumps({
        "labels_equal": True,
        "scores_max_abs_error": score_error,
        "boxes_max_abs_error_px": box_error,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
