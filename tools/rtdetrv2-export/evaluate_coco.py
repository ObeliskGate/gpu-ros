#!/usr/bin/env python3
"""Evaluate all 300 formal RT-DETRv2 candidates on COCO val2017."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torchvision.transforms.v2 as transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--val2017", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    coco = COCO(str(args.annotations))
    image_ids = sorted(coco.getImgIds())
    category_ids = sorted(coco.getCatIds())
    if len(category_ids) != 80:
        raise RuntimeError(f"expected 80 COCO categories, got {len(category_ids)}")
    session = ort.InferenceSession(str(args.onnx), providers=[args.provider])
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    predictions = []
    for image_id in image_ids:
        metadata = coco.loadImgs([image_id])[0]
        path = args.val2017 / metadata["file_name"]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = preprocess(rgb).numpy()[None, ...]
        target_sizes = np.asarray(
            [[metadata["height"], metadata["width"]]], dtype=np.int64
        )
        labels, boxes, scores = session.run(
            ["labels", "boxes", "scores"],
            {"images": tensor, "orig_target_sizes": target_sizes},
        )
        if labels.shape != (1, 300):
            raise RuntimeError(f"model did not return all 300 candidates: {labels.shape}")
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            x1, y1, x2, y2 = (float(value) for value in box)
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(category_ids[int(label)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ap = float(evaluator.stats[0] * 100.0)
    ap50 = float(evaluator.stats[1] * 100.0)
    result = {
        "image_count": len(image_ids),
        "candidates_per_image": 300,
        "AP": ap,
        "AP50": ap50,
        "official_reference": {"AP": 53.4, "AP50": 71.6},
    }
    if abs(ap - 53.4) > 0.5 or abs(ap50 - 71.6) > 0.5:
        raise AssertionError(f"COCO accuracy outside ±0.5 percentage point: {result}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
