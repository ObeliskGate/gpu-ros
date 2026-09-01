#!/usr/bin/env python3
# Copyright 2026 Boshen Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Count input messages during a bounded ROS 2 high-load run."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class MessageCounter(Node):
    """Count sensor_msgs/Image messages without storing payloads."""

    def __init__(self, topic: str, stop_file: Path | None = None):
        super().__init__('phase2b_message_counter')
        self.topic = topic
        self.stop_file = stop_file
        self.count = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.create_subscription(Image, topic, self._on_message, qos_profile_sensor_data)
        self._stop_timer = None
        if self.stop_file is not None:
            self._stop_timer = self.create_timer(0.25, self._poll_stop_file)

    def _on_message(self, _message: Image) -> None:
        self.count += 1

    def _poll_stop_file(self) -> None:
        if self.stop_file is not None and self.stop_file.exists() and rclpy.ok():
            rclpy.shutdown()

    def report(self, output_path: Path) -> None:
        report = {
            'schema_version': 1,
            'topic': self.topic,
            'message_count': self.count,
            'started_at': self.started_at,
            'finished_at': datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--stop-file')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_json)
    if output_path.exists():
        raise RuntimeError(f'refusing to overwrite existing report: {output_path}')
    rclpy.init()
    stop_file = Path(args.stop_file) if args.stop_file else None
    node = MessageCounter(args.topic, stop_file)

    def stop(_signum, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.report(output_path)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
