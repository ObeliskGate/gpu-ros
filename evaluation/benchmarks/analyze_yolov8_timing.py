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


def mean_ms(values_ns: list[int]) -> float:
    if not values_ns:
        return 0.0
    return sum(values_ns) / len(values_ns) / 1_000_000


def encoder_end_ns(record: dict[str, Any]) -> int:
    return record["callback_start_ns"] + record["total_ns"]


def handoff_ns(
    record: dict[str, Any], encoder_by_id: dict[int, dict[str, Any]]
) -> int | None:
    source = encoder_by_id.get(record["message_id"])
    if source is None:
        return None
    return record["callback_start_ns"] - encoder_end_ns(source)


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


def prior_contiguous_window(
    records: list[dict[str, Any]],
    previous_index: int,
    maximum_count: int = 32,
    maximum_callback_gap_ns: int = 100_000_000,
) -> list[dict[str, Any]]:
    """Return the records before a gap without crossing a trial boundary."""
    window_start = previous_index
    while window_start > 0 and previous_index - window_start + 1 < maximum_count:
        previous = records[window_start - 1]
        current = records[window_start]
        callback_gap_ns = current["callback_start_ns"] - previous["callback_start_ns"]
        if current["message_id"] != previous["message_id"] + 1:
            break
        if callback_gap_ns <= 0 or callback_gap_ns > maximum_callback_gap_ns:
            break
        window_start -= 1
    return records[window_start:previous_index + 1]


def main() -> None:
    args = parse_args()
    prefix = Path(args.prefix)
    encoder = read_report(Path(f"{prefix}.encoder.csv"))
    inference = read_report(Path(f"{prefix}.inference.csv"))
    interval_ms = 1000.0 / args.rate

    encoder_ok = [record for record in encoder if record["status"] == "ok"]
    inference_ok = [record for record in inference if record["status"] == "ok"]
    encoder_by_id = {record["message_id"]: record for record in encoder_ok}

    all_handoffs_ns = []
    for record in inference_ok:
        delay = handoff_ns(record, encoder_by_id)
        if delay is not None:
            all_handoffs_ns.append(delay)

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
    print_distribution(
        "inference lock wait", [record["lock_wait_ns"] for record in inference_ok]
    )
    has_stage_timing = bool(inference_ok) and "ort_session_run_ns" in inference_ok[0]
    if has_stage_timing:
        print_distribution(
            "inference input setup",
            [record["input_setup_ns"] for record in inference_ok],
        )
        print_distribution(
            "ORT session run",
            [record["ort_session_run_ns"] for record in inference_ok],
        )
        print_distribution(
            "inference output materialize",
            [record["output_materialize_ns"] for record in inference_ok],
        )
    print_distribution("encoder-to-inference handoff", all_handoffs_ns)

    gaps = find_gaps(inference_ok)
    clusters = merge_gap_clusters(gaps)
    print(f"inference ID gaps: {len(gaps)} ranges in {len(clusters)} clusters")
    for cluster_number, (start, end, previous_index) in enumerate(clusters, 1):
        window = prior_contiguous_window(inference_ok, previous_index)
        window_runs = [record["run_inference_ns"] for record in window]
        window_totals = [record["total_ns"] for record in window]
        window_handoffs = []
        for record in window:
            delay = handoff_ns(record, encoder_by_id)
            if delay is not None:
                window_handoffs.append(delay)

        arrival_intervals = []
        callback_intervals = []
        post_callback_gaps = []
        for previous, current in zip(window, window[1:]):
            if current["message_id"] != previous["message_id"] + 1:
                continue
            previous_source = encoder_by_id.get(previous["message_id"])
            current_source = encoder_by_id.get(current["message_id"])
            if previous_source is not None and current_source is not None:
                arrival_intervals.append(
                    encoder_end_ns(current_source) - encoder_end_ns(previous_source)
                )
            callback_intervals.append(
                current["callback_start_ns"] - previous["callback_start_ns"]
            )
            post_callback_gaps.append(
                current["callback_start_ns"]
                - previous["callback_start_ns"]
                - previous["total_ns"]
            )

        missing = sum(
            gap_end - gap_start + 1
            for gap_start, gap_end, _ in gaps
            if gap_start >= start and gap_end <= end
        )
        slowest = max(window, key=lambda record: record["run_inference_ns"])
        first_handoff = window_handoffs[0] if window_handoffs else 0
        last_handoff = window_handoffs[-1] if window_handoffs else 0
        print(
            f"cluster {cluster_number}: ids={start}..{end} missing={missing} "
            f"prior{len(window)}_ids={window[0]['message_id']}..{window[-1]['message_id']}"
        )
        print(
            f"  mean arrival={mean_ms(arrival_intervals):.3f}ms "
            f"callback_interval={mean_ms(callback_intervals):.3f}ms "
            f"run={mean_ms(window_runs):.3f}ms total={mean_ms(window_totals):.3f}ms"
        )
        print(
            f"  handoff={first_handoff / 1_000_000:.3f}ms"
            f"->{last_handoff / 1_000_000:.3f}ms "
            f"max_post_callback_gap={max(post_callback_gaps, default=0) / 1_000_000:.3f}ms"
        )
        print(
            f"  slowest_run=id {slowest['message_id']} "
            f"run={slowest['run_inference_ns'] / 1_000_000:.3f}ms "
            f"total={slowest['total_ns'] / 1_000_000:.3f}ms"
        )
        if has_stage_timing:
            print(
                f"  slowest phases: input_setup="
                f"{slowest['input_setup_ns'] / 1_000_000:.3f}ms "
                f"ort_session={slowest['ort_session_run_ns'] / 1_000_000:.3f}ms "
                f"output_materialize="
                f"{slowest['output_materialize_ns'] / 1_000_000:.3f}ms"
            )
            print(
                f"  prior{len(window)} phase means: input_setup="
                f"{mean_ms([record['input_setup_ns'] for record in window]):.3f}ms "
                f"ort_session="
                f"{mean_ms([record['ort_session_run_ns'] for record in window]):.3f}ms "
                f"output_materialize="
                f"{mean_ms([record['output_materialize_ns'] for record in window]):.3f}ms"
            )


if __name__ == "__main__":
    main()
