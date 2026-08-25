#!/usr/bin/env python
"""Compare the pinned PyTorch checkpoint and exported ONNX on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    pytorch_root = upstream / "rtdetrv2_pytorch"
    sys.path.insert(0, str(pytorch_root))
    from src.core import YAMLConfig  # pylint: disable=import-outside-toplevel

    official_config = (
        pytorch_root / "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    )
    with tempfile.TemporaryDirectory(prefix="rtdetrv2-validation-") as temporary:
        config = Path(temporary) / "offline_validation_config.yml"
        config.write_text(
            "__include__:\n"
            f"  - {json.dumps(str(official_config))}\n"
            "PResNet:\n"
            "  pretrained: false\n"
        )
        cfg = YAMLConfig(str(config), resume=str(args.checkpoint.resolve()))
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        model = cfg.model.deploy().eval()
        postprocessor = cfg.postprocessor.deploy()

        generator = torch.Generator(device="cpu").manual_seed(20260824)
        images = torch.rand((1, 3, 640, 640), generator=generator, dtype=torch.float32)
        target_sizes = torch.tensor([[480, 640]], dtype=torch.int64)
        with torch.no_grad():
            expected = postprocessor(model(images), target_sizes)

    session = ort.InferenceSession(
        str(args.onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        ["labels", "boxes", "scores"],
        {"images": images.numpy(), "orig_target_sizes": target_sizes.numpy()},
    )
    expected_arrays = [item.detach().cpu().numpy() for item in expected]
    np.testing.assert_array_equal(actual[0], expected_arrays[0])
    np.testing.assert_allclose(actual[1], expected_arrays[1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(actual[2], expected_arrays[2], rtol=1e-4, atol=1e-5)
    print(
        json.dumps(
            {
                "candidate_count": int(actual[0].shape[1]),
                "labels_equal": True,
                "boxes_max_abs_error": float(np.max(np.abs(actual[1] - expected_arrays[1]))),
                "scores_max_abs_error": float(np.max(np.abs(actual[2] - expected_arrays[2]))),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
