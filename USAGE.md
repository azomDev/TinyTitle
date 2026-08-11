# TinyTitle usage

how to build, train, evaluate, and compare TinyTitle. this file is the
reference for the commands in the readme.

## setup

- python 3.14+ with a virtualenv: torch, tokenizers, numpy (for training and the
  python reference).
- cc / gcc (optional musl-gcc for a static build).
- llama_cpp python bindings only needed for the supra comparison (see below).

## build the runtime

```bash
make -C runtime
# or directly:
cc -std=c11 -O3 -march=native -DNDEBUG -o runtime/title-v1 runtime/main.c -lm
```

## export the model

the trained checkpoint lives in `models/tt-v1/`. export it to the int8 `.ttm1`
format the runtime reads:

```bash
.venv/bin/python runtime/export.py --ckpt models/tt-v1/best.pt \
    --out models/tt-v1/model.ttm1
```

## generate a title

```bash
runtime/title-v1 models/tt-v1/model.ttm1 "Why does my wifi keep dropping?"
```

beam-2 is opt-in (more cpu, maybe nicer, not sure):

```bash
TTM_BEAM_WIDTH=2 runtime/title-v1 models/tt-v1/model.ttm1 "message"
```

## train from scratch

```bash
# 1. train the 8k unigram tokenizer (writes tok-8k.ttok + tokenizer-8k.json)
.venv/bin/python tools/train_tokenizer.py

# 2. train the model (20k steps, ~1.8M params)
.venv/bin/python tools/train.py train --steps 20000 --bs 64 --lr 3e-3 \
    --out models/tt-v1

# optional int8 qat fine-tune (only if the fp32/int8 gap bothers you)
.venv/bin/python tools/train.py finetune --resume models/tt-v1/best.pt \
    --out models/tt-v1-qat
```

## verify

```bash
# c/python tokenizer + decode parity (should be 9/9 and 30/30)
.venv/bin/python tools/parity.py --ttm1 models/tt-v1/model.ttm1

# quality report on the test split (rouge-l, chr-f, token f1)
.venv/bin/python tools/eval.py --ttm1 models/tt-v1/model.ttm1 --limit 200

# rss/cpu gate (must stay under 10 mib)
.venv/bin/python tools/bench.py --ttm1 models/tt-v1/model.ttm1

# everything at once
.venv/bin/python tools/gate.py --ttm1 models/tt-v1/model.ttm1
```

## compare against the supra 50M reference

the SupraLabs 50M model is used as a larger reference for a visual sanity check.
download the Q8_0 gguf once (56 MB, apache-2.0):

```bash
.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('SupraLabs/supra-title-50M-pre-gguf', \
'SupraTitle-50M-Q8_0.gguf', local_dir='models/supra-50m')"

.venv/bin/python tools/compare.py
```

runs TinyTitle and the supra model on the chat-style prompts in
`data/processed/prompts.txt` and prints a side-by-side block per prompt. point
it at any file with one prompt per line via `--prompt-file`.

## repo layout

```
tools/train_tokenizer.py   8k unigram tokenizer
tools/train.py             word-copy training
runtime/export.py          checkpoint -> .ttm1 (int8)
runtime/ttm.h              loader + tokenizer
runtime/main.c             GRU/attention/copy forward + greedy decode
tools/ref.py               python reference (parity + eval)
tools/eval.py              quality report (rouge-l, chr-f, f1)
tools/bench.py             rss/cpu gate
tools/parity.py            c/python tokenizer + decode parity
tools/compare.py           side-by-side vs the Supra 50M reference model
tools/build_dataset.py     assembles train/dev/test jsonl
tools/build_manifest.py    builds the fixed stratified human-gate manifest
data/processed/prompts.txt  the compare prompts
```
