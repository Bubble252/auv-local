"""Shared argparse parameter groups for low-level command entry points."""

from __future__ import annotations

import argparse


def add_seed_range_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the pre-registered seed range used by screening runners."""

    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
