#!/usr/bin/env python3
from __future__ import annotations

import argparse

from firefusion.example import generate_example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(generate_example(args.output, seed=args.seed).resolve())
