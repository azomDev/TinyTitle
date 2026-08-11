#!/usr/bin/env python3
"""Export a word-copy checkpoint to the TinyTitle .ttm1 runtime format.

Format sections (all little-endian):
  header:  magic "TTM1" (4) | u32 n_vocab | u32 n_sections | u64 section table offset
  section table: n x { u32 tag, u32 pad, u64 offset, u64 size }
  sections:
    TAG_VOCAB  = 1:  u32 count, then count x (u8 blen, bytes, u16 id, f32 score)
    TAG_WEIGHTS= 2:  u32 n_tensors, then per tensor:
                       u32 name_len, name bytes, u32 dims (1 or 2), i32 dim0, i32 dim1
                       then: dim==2 -> u64 n_rows, u64 row_bytes, f32 scale, int8 row...
                             dim==1 -> f32 values...
  Tensor names follow the python state_dict keys exactly (no prefix).
  All tensors are per-row int8 with a f32 scale; 1-D tensors are raw f32.
  The magic identifies the format version; no separate metadata section is stored.

Usage:
    .venv/bin/python runtime/export.py --ckpt models/tt-v1/best.pt --out models/tt-v1/model.ttm1
"""
import argparse
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from train import PointerGenerator, load_vocab, DEFAULT_TOK  # noqa: E402

MAGIC = b"TTM1"
TAG_VOCAB = 1
TAG_WEIGHTS = 2


def quantize_row(w):
    """Per-row symmetric int8: returns (scale_f32, int8_bytes)."""
    w = w.detach().float()
    out, inp = w.shape
    scales = []
    rows = []
    for r in range(out):
        row = w[r]
        amax = row.abs().max().clamp_min(1e-8)
        scale = amax / 127.0
        q = torch.clamp(torch.round(row / scale), -127, 127).to(torch.int8)
        scales.append(scale.item())
        rows.append(q.numpy().tobytes())
    return struct.pack(f"<{out}f", *scales), b"".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--f32", action="store_true",
                    help="export raw f32 weights (reference; default is per-row int8)")
    args = ap.parse_args()

    vocab = load_vocab(args.tokenizer)
    V = len(vocab)
    model = PointerGenerator(V)
    ck = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ck, dict) or ck.get("objective") != "word-copy":
        raise ValueError("incompatible checkpoint: exporter requires word-copy")
    model.load_state_dict(ck["model"])
    model.eval()

    tensors = []
    for name, p in model.named_parameters():
        if name == "head.weight":
            continue  # tied to emb.weight
        w = p.detach().float()
        tensors.append((name, w))

    # --- vocab section ---
    # { u32 count, then per token: u8 blen, bytes, u16 unused, f32 score }
    # scores come from the tokens-lib json so C and python viterbi agree exactly.
    tok_json = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "tokenizer-8k.json")
    with open(tok_json) as f:
        tdata = json.load(f)
    tvocab = tdata["model"]["vocab"]  # [[token, score], ...]
    assert len(tvocab) == V, (len(tvocab), V)
    scores = {t: float(s) for t, s in tvocab}
    vocab_bytes = bytearray()
    vocab_bytes += struct.pack("<I", V)
    for t in vocab:
        b = t.encode("utf-8")
        assert len(b) <= 255
        sc = scores.get(t, -20.0)
        vocab_bytes += bytes([len(b)]) + b + struct.pack("<H", 0) + struct.pack("<f", sc)
    vocab_size = len(vocab_bytes)

    # --- weights section ---
    wbytes = bytearray()
    wbytes += struct.pack("<I", len(tensors))
    for name, w in tensors:
        nb = name.encode("utf-8")
        wbytes += struct.pack("<I", len(nb)) + nb
        # pad the name to 4-byte alignment so tensor data is 4-aligned
        while len(wbytes) % 4 != 0:
            wbytes += b"\x00"
        wbytes += struct.pack("<I", w.dim())
        if w.dim() == 2:
            wbytes += struct.pack("<ii", w.shape[0], w.shape[1])
            if args.f32:
                wbytes += struct.pack("<Q", w.shape[0])
                wbytes += struct.pack("<Q", 0)  # row_bytes=0 marks f32
                wbytes += w.numpy().astype("float32").tobytes()
            else:
                scales, rows = quantize_row(w)
                wbytes += struct.pack("<Q", w.shape[0])
                wbytes += struct.pack("<Q", len(rows))
                wbytes += scales + rows
        else:
            wbytes += struct.pack("<ii", w.shape[0], 0)
            wbytes += w.numpy().astype("float32").tobytes()
    weights_size = len(wbytes)

    # --- file layout (pad sections to 8-byte alignment) ---
    header_size = 4 + 4 + 4 + 8  # magic, n_vocab, n_sections, table_off
    table_size = 2 * 24
    off = header_size + table_size

    def align8(x):
        return (x + 7) & ~7

    sections = [
        (TAG_VOCAB, off, vocab_size),
        (TAG_WEIGHTS, align8(off + vocab_size), weights_size),
    ]
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", V))
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", header_size))
        for tag, o, sz in sections:
            f.write(struct.pack("<I", tag) + struct.pack("<I", 0) + struct.pack("<Q", o) + struct.pack("<Q", sz))
        # pad to first section
        f.write(bytes(sections[0][1] - f.tell()))
        f.write(bytes(vocab_bytes))
        f.write(bytes(sections[1][1] - f.tell()))
        f.write(bytes(wbytes))
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"  tensors: {len(tensors)}  vocab: {V}")


if __name__ == "__main__":
    main()
