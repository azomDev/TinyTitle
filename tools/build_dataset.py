#!/usr/bin/env python3
"""Phase 1: assemble + filter + dedupe the chat-title training data.

Reads three public datasets, normalizes to (message, title) pairs, filters
(English, sane title lengths, bad-title patterns), dedupes on message,
splits train/dev/test, writes JSONL + a summary.

Usage:
    .venv/bin/python tools/build_dataset.py
"""

import json
import os
import re
import unicodedata

from datasets import load_dataset

DATASETS = [
    # (hf name, message col, title col)
    ("SupraLabs/chat-titles-filtered-115K", "user", "title", "supra-filtered"),
    ("ogrnz/chat-titles", "message", "title", "ogrnz"),
    ("Michionlion/chat-titles-english", "user", "title", "michionlion"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

MIN_MSG_CHARS = 20
MAX_MSG_CHARS = 6000
MIN_TITLE_WORDS = 2
MAX_TITLE_WORDS = 15
MAX_TITLE_CHARS = 80

# patterns that make a title useless for our task (JSON-y, meta, placeholders)
BAD_TITLE_RE = re.compile(
    r"(\{\{|\}\}|<[a-z]+>|placeholder|untitled|n/a|none|todo|fixme|"
    r"^[\s\W_]+$|lorem ipsum|example title)",
    re.IGNORECASE,
)

# messages that are clearly benchmark/template junk
BAD_MSG_RE = re.compile(
    r"(complete the sentence|fill in the blank|multiple choice|"
    r"you will be given a definition of a task|sentence to rdf|"
    r"translate the following|paraphrase the|rewrite the sentence)",
    re.IGNORECASE,
)


EN_STOP = set(
    "the a an and or but of to in on for with by from as at is are was were be been being "
    "it its this that these those i you he she we they me my your his her our their "
    "have has had do does did can could will would should may might not no yes "
    "about into over under what which who whom when where why how if then than so "
    "um der die das und zu von mit fur auf ein eine ist sind war fur den dem des "
    "je tu il elle nous vous ils elles le la les un une et ou mais des pour dans "
    "como que para por con sin es son fue eran está está no sí pero"
    .split()
)

# German/French/Spanish function words that should NOT appear in English text
FOREIGN_STOP = set(
    "der die das und zu von mit fur auf ein eine ist sind war den dem des nicht "
    "pour dans avec une des les est sont un la le et ou mais ce cette ces "
    "como que para por con sin es son fue eran está pero y o el la los las"
    .split()
)


def is_english(s):
    """Strict-ish heuristic: reject non-English (diacritics OR foreign stopwords)."""
    if not s:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) >= 0.05:
        return False
    words = re.findall(r"[a-zA-Z]+", s.lower())
    if not words:
        return False
    foreign = sum(1 for w in words if w in FOREIGN_STOP)
    # drop if ANY French/German/Spanish function word appears (strict for translation input)
    return foreign == 0


def clean_title(t):
    if not t:
        return None
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,;:!?\"'")
    if not t:
        return None
    if len(t) > MAX_TITLE_CHARS:
        return None
    return t


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    seen = set()
    pairs = []  # (source, message, title)
    stats = {name: {"read": 0, "kept": 0, "dup": 0} for _, _, _, name in DATASETS}

    for hf_name, msg_col, title_col, src in DATASETS:
        print(f"\n=== {hf_name} ===", flush=True)
        ds = load_dataset(hf_name, split="train", streaming=True)
        for i, row in enumerate(ds):
            msg = (row.get(msg_col) or "").strip()
            title = clean_title(row.get(title_col) or "")
            stats[src]["read"] += 1

            if (
                len(msg) < MIN_MSG_CHARS
                or len(msg) > MAX_MSG_CHARS
                or not title
                or not is_english(msg)
                or not is_english(title)
                or BAD_MSG_RE.search(msg)
                or BAD_TITLE_RE.search(title)
            ):
                continue

            words = title.split()
            if not (MIN_TITLE_WORDS <= len(words) <= MAX_TITLE_WORDS):
                continue

            key = msg.strip().lower()
            if key in seen:
                stats[src]["dup"] += 1
                continue
            seen.add(key)
            stats[src]["kept"] += 1
            pairs.append((src, msg, title))

            if i and i % 20000 == 0:
                print(f"  read {i:,} kept {stats[src]['kept']:,}", flush=True)

    print("\n=== summary ===")
    for src, s in stats.items():
        print(f"  {src}: read {s['read']:,} kept {s['kept']:,} dup {s['dup']:,}")

    # shuffle deterministically
    import random

    rng = random.Random(42)
    rng.shuffle(pairs)

    # splits: test = 2k, dev = 2k, rest train
    test = pairs[:2000]
    dev = pairs[2000:4000]
    train = pairs[4000:]

    def write(name, items):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            for src, msg, title in items:
                f.write(json.dumps({"source": src, "message": msg, "title": title}) + "\n")
        print(f"  wrote {name}: {len(items):,} pairs")

    write("test.jsonl", test)
    write("dev.jsonl", dev)
    write("train.jsonl", train)

    print("\ndone.")


if __name__ == "__main__":
    main()
