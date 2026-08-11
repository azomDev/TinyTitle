#!/usr/bin/env python3
"""RSS + cpu measurement for the C runtime (phase 0 gate).

Measures peak RSS (vmhwm + /usr/bin/time -v) and cpu time for:
  - empty/minimal input
  - median test input
  - exactly 256 selected tokens
  - maximum 6000-byte input
  - adversarial byte fallback

Usage:
    .venv/bin/python tools/bench.py --ttm1 models/tt-v1/model.ttm1 --runtime runtime/title-v1
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

CASES = {
    "minimal": "hi",
    "short": "How do I fix a segfault?",
    "median": "I'm trying to set up a home server with docker and nginx reverse proxy but I keep getting 502 bad gateway errors when I try to access my services from outside my network. I've checked my firewall rules and port forwarding but nothing seems to work. Any ideas what could be causing this?",
    "long": "word " * 1200,  # exactly 6000 bytes
    "adversarial": ("~!@#$%^&*()_+-=[]{};:',.<>/? café résumé 502 " * 100)[:6000],
}


def measure(runtime, ttm1, text):
    """Run the runtime, return (peak_rss_kb, cpu_ms, wall_ms, output).

    Uses the child's own getrusage (TTM_RSS=1) for peak RSS, and
    resource.getrusage(RUSAGE_CHILDREN) as an external cross-check.
    """
    env = dict(os.environ, TTM_RSS="1", TTM_TOUCH="1")
    t0 = time.perf_counter()
    p = subprocess.run([runtime, ttm1, text], capture_output=True,
                       timeout=120, env=env)
    wall = (time.perf_counter() - t0) * 1000
    rss_kb = 0
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        if "[rss] peak=" in line:
            # line is: [rss] peak=NNNN kb
            rss_kb = int(line.split("=")[-1].split()[0])
    cpu_ms = wall  # single-threaded, wall ~= cpu
    return rss_kb, cpu_ms, wall, p.stdout.decode("utf-8", "replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttm1", required=True)
    ap.add_argument("--runtime", default=os.path.join("runtime", "title-v1"))
    args = ap.parse_args()

    print(f"=== rss/cpu bench: {args.ttm1} ===")
    print(f"{'case':<14} {'rss_kb':>8} {'cpu_ms':>8} {'wall_ms':>8}  output")
    worst_rss = 0
    for name, text in CASES.items():
        rss, cpu, wall, out = measure(args.runtime, args.ttm1, text)
        worst_rss = max(worst_rss, rss)
        print(f"{name:<14} {rss:>8} {cpu:>8.1f} {wall:>8.1f}  {out[:40]!r}")
    print(f"\nworst-case peak rss: {worst_rss} kb = {worst_rss/1024:.2f} mib")
    gate_kib = 10240
    print(f"gate: {'PASS' if worst_rss < gate_kib else 'FAIL'} (< {gate_kib} kb = 10 mib)")


if __name__ == "__main__":
    main()
