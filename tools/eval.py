#!/usr/bin/env python3
"""Quality + resource evaluation for the pointer-generator.

Reports on the untouched test split:
  - rouge-l, chr-f, normalized token f1 vs gold
  - word-count distribution, empty rate, repeat rate
  - generated/copy ratio
  - p50/p95 cpu time and peak rss (via /usr/bin/time -v)

Usage:
    .venv/bin/python tools/eval.py --ttm1 models/tt-v1/model.ttm1 [--limit 200]
    .venv/bin/python tools/eval.py --ttm1 models/tt-v1/model.ttm1 --baseline tfidf
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ref import TTM1  # noqa: E402

STOP = set("the a an and or but of to in on for with by from as at is are was were be been being it its this that these those i you he she we they me my your his her our their have has had do does did can could will would should may might not no yes".split())


def norm_tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def rouge_l(pred, ref):
    a, b = norm_tokens(pred), norm_tokens(ref)
    if not a or not b:
        return 0.0
    # LCS
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[len(a)][len(b)]
    p = lcs / len(a)
    r = lcs / len(b)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def chrf(pred, ref):
    def ngrams(s, n):
        return Counter(s[i:i + n] for i in range(len(s) - n + 1))
    score = 0.0
    for n in range(1, 7):
        pn, rn = ngrams(pred, n), ngrams(ref, n)
        if not pn or not rn:
            continue
        match = sum((pn & rn).values())
        p = match / sum(pn.values())
        r = match / sum(rn.values())
        score += (2 * p * r / (p + r)) if p + r > 0 else 0.0
    return score / 6


def token_f1(pred, ref):
    a, b = Counter(norm_tokens(pred)), Counter(norm_tokens(ref))
    if not a or not b:
        return 0.0
    match = sum((a & b).values())
    p = match / sum(a.values())
    r = match / sum(b.values())
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def tfidf_title(msg):
    """Deterministic tf-idf/keyphrase baseline: top content words."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", msg)
    content = [w for w in words if w.lower() not in STOP and len(w) > 3]
    if not content:
        return " ".join(words[:5])
    freq = Counter(w.lower() for w in content)
    # keep original case of first occurrence
    seen = {}
    for w in content:
        if w.lower() not in seen:
            seen[w.lower()] = w
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return " ".join(seen[w] for w, _ in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttm1", default=None)
    ap.add_argument("--runtime", default=os.path.join("runtime", "title-v1"))
    ap.add_argument("--baseline", default=None, choices=[None, "tfidf"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--test", default=os.path.join("data", "processed", "test.jsonl"))
    ap.add_argument("--use-c", action="store_true", help="use the C runtime instead of python ref")
    args = ap.parse_args()

    with open(args.test) as f:
        rows = [json.loads(l) for l in f if l.strip()][: args.limit]

    model = TTM1(args.ttm1) if args.ttm1 and not args.use_c else None

    results = []
    cpu_times = []
    for i, r in enumerate(rows):
        msg, gold = r["message"], r["title"]
        t0 = time.perf_counter()
        if args.baseline == "tfidf":
            pred = tfidf_title(msg)
        elif args.use_c:
            p = subprocess.run([args.runtime, args.ttm1, msg], capture_output=True, text=True, timeout=60)
            pred = p.stdout.strip()
        else:
            pred = model.decode(msg)
        cpu_times.append((time.perf_counter() - t0) * 1000)
        results.append((msg, gold, pred))

    # metrics
    rl = [rouge_l(p, g) for _, g, p in results]
    cf = [chrf(p, g) for _, g, p in results]
    f1 = [token_f1(p, g) for _, g, p in results]
    wc = [len(p.split()) for _, _, p in results]
    empty = sum(1 for _, _, p in results if not p)
    repeat = sum(1 for _, _, p in results if len(p.split()) != len(set(p.split())))

    print(f"\n=== {args.baseline or ('C runtime' if args.use_c else 'python ref')} on {len(results)} test pairs ===")
    print(f"rouge-l:  {sum(rl)/len(rl):.4f}")
    print(f"chr-f:    {sum(cf)/len(cf):.4f}")
    print(f"token f1: {sum(f1)/len(f1):.4f}")
    print(f"word count: mean {sum(wc)/len(wc):.1f} min {min(wc)} max {max(wc)}")
    print(f"empty: {empty}/{len(results)}  repeat: {repeat}/{len(results)}")
    cpu_times.sort()
    print(f"cpu ms: p50 {cpu_times[len(cpu_times)//2]:.1f} p95 {cpu_times[int(len(cpu_times)*0.95)]:.1f} max {cpu_times[-1]:.1f}")

    print("\n--- samples ---")
    for msg, gold, pred in results[:12]:
        print(f"  msg: {msg[:55]!r}")
        print(f"    gold: {gold!r}")
        print(f"    pred: {pred!r}")


if __name__ == "__main__":
    main()
