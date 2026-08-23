#!/usr/bin/env python
# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize low-overhead YOLOv8 encoder and inference timing reports."""

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prefix",
        help="Report prefix used by YOLOV8_BENCHMARK_TIMING_PREFIX",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=188.3125,
        help="Measured input rate in Hz (default: 188.3125)",
    )
    return parser.parse_args()


def read_report(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for key, value in tuple(record.items()):
            if key != "status":
                record[key] = int(value)
    return records


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def print_distribution(label: str, values_ns: list[int]) -> None:
    values_ms = [value / 1_000_000 for value in values_ns]
    print(
        f"{label}: p50={percentile(values_ms, 0.50):.3f}ms "
        f"p95={percentile(values_ms, 0.95):.3f}ms "
        f"p99={percentile(values_ms, 0.99):.3f}ms "
        f"max={max(values_ms, default=0.0):.3f}ms"
    )


def find_gaps(records: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    gaps = []
    previous_id = None
    previous_index = None
    for index, record in enumerate(records):
        message_id = record["message_id"]
        if previous_id is not None and message_id > previous_id + 1:
            gaps.append((previous_id + 1, message_id - 1, previous_index))
        previous_id = message_id
        previous_index = index
    return gaps


def merge_gap_clusters(
    gaps: list[tuple[int, int, int]], maximum_separation: int = 20
) -> list[tuple[int, int, int]]:
    clusters = []
    for start, end, previous_index in gaps:
        if clusters and start <= clusters[-1][1] + maximum_separation:
            clusters[-1] = (clusters[-1][0], end, clusters[-1][2])
        else:
            clusters.append((start, end, previous_index))
    return clusters


def main() -> None:
    args = parse_args()
    prefix = Path(args.prefix)
    encoder = read_report(Path(f"{prefix}.encoder.csv"))
    inference = read_report(Path(f"{prefix}.inference.csv"))
    interval_ms = 1000.0 / args.rate

    encoder_ok = [record for record in encoder if record["status"] == "ok"]
    inference_ok = [record for record in inference if record["status"] == "ok"]
    encoder_by_id = {record["message_id"]: record for record in encoder_ok}

    handoff_ns = []
    for record in inference_ok:
        source = encoder_by_id.get(record["message_id"])
        if source is not None:
            encoder_end = source["callback_start_ns"] + source["total_ns"]
            handoff_ns.append(record["callback_start_ns"] - encoder_end)

    print(f"target interval: {interval_ms:.3f}ms at {args.rate:.4f}Hz")
    print(f"encoder callbacks: {len(encoder)} ({len(encoder_ok)} ok)")
    print(f"inference callbacks: {len(inference)} ({len(inference_ok)} ok)")
    print_distribution("encoder encode", [record["encode_ns"] for record in encoder_ok])
    print_distribution("encoder publish", [record["publish_ns"] for record in encoder_ok])
    print_distribution(
        "inference run", [record["run_inference_ns"] for record in inference_ok]
    )
    print_distribution(
        "inference publish", [record["publish_ns"] for record in inference_ok]
    )
    print_distribution("encoder-to-inference handoff", handoff_ns)

    gaps = find_gaps(inference_ok)
    clusters = merge_gap_clusters(gaps)
    print(f"inference ID gaps: {len(gaps)} ranges in {len(clusters)} clusters")
    for cluster_number, (start, end, previous_index) in enumerate(clusters, 1):
        window_start = max(0, previous_index - 9)
        window = inference_ok[window_start:previous_index + 1]
        window_runs = [record["run_inference_ns"] for record in window]
        window_totals = [record["total_ns"] for record in window]
        window_handoffs = []
        for record in window:
            source = encoder_by_id.get(record["message_id"])
            if source is not None:
                encoder_end = source["callback_start_ns"] + source["total_ns"]
                window_handoffs.append(record["callback_start_ns"] - encoder_end)
        missing = sum(gap_end - gap_start + 1 for gap_start, gap_end, _ in gaps
                      if gap_start >= start and gap_end <= end)
        print(
            f"cluster {cluster_number}: ids={start}..{end} missing={missing} "
            f"prior10_max_run={max(window_runs, default=0) / 1_000_000:.3f}ms "
            f"prior10_max_total={max(window_totals, default=0) / 1_000_000:.3f}ms "
            f"prior10_max_handoff={max(window_handoffs, default=0) / 1_000_000:.3f}ms"
        )


if __name__ == "__main__":
    main()
