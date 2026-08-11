#!/usr/bin/env python3
"""Build the fixed, stratified human-gate manifest (300 messages).

Strata: conversational, code, rare identifiers, long prompts, requests.
Deterministic (seed 42), drawn from the untouched test split.
Output: data/processed/manifest.jsonl
"""
import json
import os
import random
import re

TEST = os.path.join("data", "processed", "test.jsonl")
OUT = os.path.join("data", "processed", "manifest.jsonl")


def classify(msg):
    if re.search(r"(def |import |SELECT |\{\{|=>|```|\bint\b|\bchar\b|:\s*$)", msg):
        return "code"
    if len(msg.split()) > 150:
        return "long"
    if re.search(r"\b[A-Za-z]{3,}_[A-Za-z0-9_]{3,}\b|error|exception|segfault|EADDR|502|404", msg):
        return "rare-id"
    if re.search(r"\?$", msg.strip()):
        return "request"
    return "conversational"


def main():
    with open(TEST) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    buckets = {}
    for r in rows:
        c = classify(r["message"])
        buckets.setdefault(c, []).append(r)
    rng = random.Random(42)
    out = []
    for c in ["conversational", "request", "code", "rare-id", "long"]:
        pool = buckets.get(c, [])
        rng.shuffle(pool)
        n = 100 if c == "conversational" else 50
        out.extend(pool[:n])
    rng.shuffle(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}: {len(out)} messages")
    from collections import Counter
    print(Counter(classify(r["message"]) for r in out))


if __name__ == "__main__":
    main()
