#!/usr/bin/env python3
"""Compare TinyTitle against the SupraLabs 50M title model on the same prompts.

Runs the TinyTitle C runtime (title-v1) and the Supra 50M gguf (llama.cpp) on
the same prompts and prints a markdown block per prompt for visual comparison.

Usage:
    .venv/bin/python tools/compare.py [--prompts N] [--ttm1 models/tt-v1/model.ttm1]
        [--runtime runtime/title-v1] [--supra models/supra-50m/SupraTitle-50M-Q8_0.gguf]

Prompts come from data/processed/prompts.txt by default (chat-style); use
--prompt-file to point at another file.
"""
import argparse
import json
import os
import subprocess
import sys

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEFAULT_PROMPTS = os.path.join(PROC_DIR, "prompts.txt")

DEFAULT_TTM1 = os.path.join("models", "tt-v1", "model.ttm1")
DEFAULT_RUNTIME = os.path.join("runtime", "title-v1")
DEFAULT_SUPRA = os.path.join("models", "supra-50m", "SupraTitle-50M-Q8_0.gguf")

GEN_KW = dict(
    temperature=0.4,
    top_p=0.85,
    top_k=40,
    repeat_penalty=1.2,
    max_tokens=24,
)


def short_prompt(msg, limit=80):
    msg = msg.replace("\n", " ")
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


def tiny_title(runtime, ttm1, msg):
    p = subprocess.run([runtime, ttm1, msg], capture_output=True, text=True, timeout=60)
    return p.stdout.strip()


def supra_title(llm, msg):
    out = llm(f"User: {msg}\nTitle: ", **GEN_KW)
    return out["choices"][0]["text"].strip().replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--ttm1", default=DEFAULT_TTM1)
    ap.add_argument("--runtime", default=DEFAULT_RUNTIME)
    ap.add_argument("--supra", default=DEFAULT_SUPRA)
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPTS,
                    help="file with one prompt per line (default data/processed/prompts.txt)")
    args = ap.parse_args()

    if not os.path.exists(args.supra):
        sys.exit(f"supra gguf not found: {args.supra}\n"
                 f"download it first:\n"
                 f"  .venv/bin/python -c \"from huggingface_hub import hf_hub_download; "
                 f"hf_hub_download('SupraLabs/supra-title-50M-pre-gguf', "
                 f"'SupraTitle-50M-Q8_0.gguf', local_dir='models/supra-50m')\"")

    from llama_cpp import Llama  # imported lazily so the script still works without llama_cpp

    with open(args.prompt_file) as f:
        prompts = [l.strip() for l in f if l.strip()]
    prompts = prompts[: args.prompts]

    llm = Llama(model_path=args.supra, n_ctx=2048, n_threads=16, verbose=False)

    for i, msg in enumerate(prompts, 1):
        tiny = tiny_title(args.runtime, args.ttm1, msg)
        supra = supra_title(llm, msg)
        print(f"### {i}. {short_prompt(msg)}\n")
        print("| model | title |")
        print("|---|---|")
        print(f"| TinyTitle | {tiny} |")
        print(f"| Supra 50M | {supra} |")
        print()


if __name__ == "__main__":
    main()
