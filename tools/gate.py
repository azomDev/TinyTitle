#!/usr/bin/env python3
"""One-command gate report: quality + rss + cpu (phase 0 exit condition).

Usage:
    .venv/bin/python tools/gate.py --ttm1 models/tt-v1/model.ttm1 --runtime runtime/title-v1
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttm1", required=True)
    ap.add_argument("--runtime", default=os.path.join("runtime", "title-v1"))
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    print(f"=== TinyTitle gate report: {args.ttm1} ===")
    print(f"model file: {os.path.getsize(args.ttm1)} bytes "
          f"({os.path.getsize(args.ttm1)/1e6:.2f} MB)")

    # quality
    print("\n--- quality (test split) ---")
    subprocess.run([sys.executable, os.path.join("tools", "eval.py"),
                    "--ttm1", args.ttm1, "--runtime", args.runtime,
                    "--use-c", "--limit", str(args.limit)],
                   check=True)

    # rss + cpu
    print("\n--- rss/cpu ---")
    subprocess.run([sys.executable, os.path.join("tools", "bench.py"),
                    "--ttm1", args.ttm1, "--runtime", args.runtime], check=True)


if __name__ == "__main__":
    main()
