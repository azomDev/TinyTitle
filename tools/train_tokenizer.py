#!/usr/bin/env python3
"""Train the 8k unigram tokenizer for the TinyTitle pointer-generator (english only).

Outputs two artifacts:
  - data/processed/tok-8k.ttok        runtime/export format (no scores)
  - data/processed/tokenizer-8k.json  tokens-lib format for python training

Both are derived from the SAME trained unigram; the runtime format must be
regenerated from the json (with scores) by export.py so C and python do
identical viterbi.
"""
import json
import os
import sys

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

from tokenizers.trainers import UnigramTrainer

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_TTOK = os.path.join(PROC_DIR, "tok-8k.ttok")
OUT_JSON = os.path.join(PROC_DIR, "tokenizer-8k.json")

VOCAB = 8000
SPECIAL = ["<pad>", "<eos>", "<unk>"]


def lines():
    # tokenizer training is model training: never leak dev/test text into it.
    path = os.path.join(PROC_DIR, "train.jsonl")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield d["message"]
            yield d["title"]


def main():
    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
    trainer = UnigramTrainer(
        vocab_size=VOCAB,
        special_tokens=SPECIAL,
        unk_token="<unk>",
        initial_alphabet=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
                          "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                          "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6",
                          "7", "8", "9"],
    )
    tok.train_from_iterator(lines(), trainer=trainer)

    vocab = tok.get_vocab()
    ids = sorted(vocab.items(), key=lambda kv: kv[1])
    if vocab.get("<pad>") != 0 or vocab.get("<eos>") != 1 or vocab.get("<unk>") != 2:
        print("error: special ids not 0/1/2", file=sys.stderr)
        sys.exit(1)
    print(f"vocab size {len(ids)} (pad=0 eos=1 unk=2)")

    tok.save(OUT_JSON)
    print(f"saved {OUT_JSON} ({os.path.getsize(OUT_JSON)} bytes)")

    # runtime .ttok: blen-prefixed strings + ids (no scores — export adds them)
    with open(OUT_TTOK, "wb") as f:
        f.write(b"TTOK1")
        f.write((len(ids)).to_bytes(4, "little"))
        for t, i in ids:
            b = t.encode("utf-8")
            f.write(bytes([len(b)]))
            f.write(b)
            f.write(i.to_bytes(2, "little"))
    print(f"saved {OUT_TTOK} ({os.path.getsize(OUT_TTOK)} bytes)")

    # sanity
    for s in ["Hello world", "Debugging a segfault in C", "Why does my WiFi keep disconnecting",
              "How to make a discord server", "Explain the difference between TCP and UDP"]:
        enc = tok.encode(s)
        print(f"  {s!r} -> {enc.tokens}")


if __name__ == "__main__":
    main()
