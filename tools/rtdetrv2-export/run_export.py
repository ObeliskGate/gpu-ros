#!/usr/bin/env python
"""Seed the pinned upstream exporter without modifying upstream source."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


def reject_network_weight_fetch(*args, **kwargs):  # noqa: ANN002, ANN003
    del args, kwargs
    raise RuntimeError(
        "the locked RT-DETRv2 export recipe forbids network weight downloads"
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-script", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.hub.load_state_dict_from_url = reject_network_weight_fetch
    sys.argv = [str(known.upstream_script), *remaining]
    runpy.run_path(str(known.upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
