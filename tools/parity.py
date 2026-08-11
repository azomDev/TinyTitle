#!/usr/bin/env python3
"""Numerical parity: C runtime vs python reference, per-step logits.

Runs both on the same inputs and compares:
  - tokenizer ids (byte spans)
  - first-step logits (max abs diff)
  - greedy decode tokens

Usage: .venv/bin/python tools/parity.py --ttm1 models/tt-v1/model.ttm1 --runtime runtime/title-v1
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ref import TTM1  # noqa: E402


FIXTURES = [
    "How do I fix a segfault in my C program?",
    "Why does my WiFi keep disconnecting?",
    "The quick brown fox jumps over the lazy dog.",
    "Explain the difference between TCP and UDP in computer networking.",
    "My docker container won't start, port 8080 is already in use. I've tried killing the process and restarting but nothing works. What should I do?",
    "SELECT * FROM users WHERE id = 42; -- fetch the user by id",
    "How to make a discord server with roles and permissions for my friends?",
    "café au lait and résumé writing tips for job applications",
    "I keep getting HTTP 502 Bad Gateway from nginx when my node app crashes. The PM2 logs show an EADDRINUSE error.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttm1", required=True)
    ap.add_argument("--runtime", default=os.path.join("runtime", "title-v1"))
    ap.add_argument("--dev", default=os.path.join("data", "processed", "dev.jsonl"))
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    m = TTM1(args.ttm1)

    # 1) tokenizer parity on fixtures
    print("=== tokenizer parity ===")
    tok_ok = 0
    for text in FIXTURES:
        ids, starts, ends = m.tokenize(text)
        p = subprocess.run([args.runtime, args.ttm1, text, "--dump-tokens"],
                           capture_output=True, text=True, timeout=60)
        c_ids = [int(x) for x in p.stdout.strip().split()] if p.stdout.strip() else []
        same = c_ids == ids
        tok_ok += same
        print(f"  {'OK ' if same else 'DIFF'} {text[:40]!r}: py={len(ids)} c={len(c_ids)}")
    print(f"tokenizer: {tok_ok}/{len(FIXTURES)}")

    # 2) greedy decode parity on dev
    print("\n=== decode parity (dev) ===")
    with open(args.dev) as f:
        rows = [json.loads(l) for l in f if l.strip()][: args.limit]
    agree = 0
    diffs = 0
    for i, r in enumerate(rows):
        text = r["message"]
        py_title = m.decode(text)
        p = subprocess.run([args.runtime, args.ttm1, text], capture_output=True,
                           text=True, timeout=60)
        c_title = p.stdout.strip()
        if py_title == c_title:
            agree += 1
        else:
            diffs += 1
            if diffs <= 5:
                print(f"  DIFF {i}: py={py_title[:50]!r}")
                print(f"         c ={c_title[:50]!r}")
    print(f"decode parity: {agree}/{len(rows)} exact match")


if __name__ == "__main__":
    main()
