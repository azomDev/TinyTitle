# TinyTitle

Tiny, CPU-driven model that turns an LLM chat message into a short title (2-10 words). English only. The whole thing (model + tokenizer + runtime) fits in under 5 mib of ram and runs in a few tens of milliseconds on a normal desktop cpu.

It is a small attentive GRU with a copy mechanism: for every title word it either generates a subword from its 8k vocabulary or copies a whole word straight out of your message, so it keeps your casing and spelling, and it never splices words into garbage like `rafe` or `sritual` (do not ask where those come from).

honestly, this project started as "can we have a title model smaller than the Supra Title 50M one on huggingface and still have it make sense". the answer is: mostly, and it is genuinely tiny. the quality ceiling is low (it is a 1.8M-param gru, not a real llm), but for the one job it does, it works decently.

i did not build any of this, to be clear. this is the kind of project that started as "let me see if this can even be done" and then I got an LLM to write nearly all the code while i nodded along and asked for changes. i barely know what is going on in here, but it is smol and works decently ¯\\\_(ツ)\_/¯

## the numbers

| metric | value |
|---|---|
| params | ~1.8M (int8) |
| model file | ~1.98 MB (`.ttm1`) |
| peak rss (ram) | 4.89 mib (all pages touched) |
| typical cpu | ~25 ms, p95 ~75 ms |
| vocab | 8k unigram (byte fallback) |
| source cap | 256 tokens (first 192 + salience 64) |

the 10 mib rss budget is the whole point. there is no transformer, no kv cache, no llama.cpp, no python at runtime. just one small C binary that mmaps the model and leaves.

## how it works

```
message
  -> metaspace unigram tokenization (byte spans tracked)
  -> pick 256 source tokens (first 192 + salience 64)
  -> group tokens into lexical words
  -> bidirectional GRU encoder
  -> attentive GRU decoder
  -> each step: generate_subword | copy_source_word | eos
  -> greedy decode
  -> drop duplicate words, title case
```

for a title word, "copy" means: find the same word in the message (after case folding) and paste it back with its original casing and any `c++`, `node.js`, `gpt-4`, `nielsen's` weirdness intact. if it is not in the message, the model spells it out from its vocabulary instead.

there is also an opt-in beam-2 decoder (`TTM_BEAM_WIDTH=2`), which was built because it sounded like the right thing to do. on the easy prompts it does not really change much, and it costs about 1.6x the cpu. so, you know. it is there if you want it.

## vs a 50M reference model

it is fun (and honest) to look at it next to the SupraLabs 50M title model, a 26x bigger model trained on much more data:

### 1. How does AI work?

| model | title |
|---|---|
| TinyTitle | AI Work |
| Supra 50M | AI Basics Explained |

### 2. Why does my WiFi keeps disconnecting the whole time??

| model | title |
|---|---|
| TinyTitle | WiFi Time Disconnecting |
| Supra 50M | WiFi disconnecting Time |

### 3. How to make a discord server?

| model | title |
|---|---|
| TinyTitle | Discord Server |
| Supra 50M | Discord Server Creation |

### 4. Can you explain the difference between TCP and UDP?

| model | title |
|---|---|
| TinyTitle | TCP and UDP Differences |
| Supra 50M | TCP Vs UDP Comparison |

### 5. I need help debugging this python script that crashes on import

| model | title |
|---|---|
| TinyTitle | Python Import Script |
| Supra 50M | Debugging Python Script |

### 6. What's the best way to learn French quickly?

| model | title |
|---|---|
| TinyTitle | Best Way to French Quickly |
| Supra 50M | Learning French Tips |

### 7. My docker container won't start, port 8080 is already in use

| model | title |
|---|---|
| TinyTitle | Docker Container 8080 |
| Supra 50M | Docker Container Port 8080 |

### 8. Explain quantum computing like I'm five

| model | title |
|---|---|
| TinyTitle | Quantum Computing Like |
| Supra 50M | Quantum Computing Basics |

roughly: the big model is more abstract and grammatical, the tiny one is more literal and sometimes drops a function word. both get the point across.

### the ram comparison

for fairness (kind of unfair since I used Q8 for the examples above but whatever), let's compare the most aggressive quantization of the 50M model that exists (Q1_0, 19.6 MB)

| | model file | peak rss (including runtime) |
|---|---|---|
| TinyTitle | 1.98 MB | 4.89 mib |
| Supra 50M (Q1_0, via llama.cpp) | 19.6 MB | ~126 mib |

so even at its smallest, the 50M model needs roughly 25x the ram of TinyTitle.

## more

- [USAGE.md](USAGE.md): build, train, verify, compare, and repo layout.
- [DESIGN.md](DESIGN.md): the actual design and correctness invariants.
- model format: `.ttm1`, magic `TTM1`. future incompatible formats increment the number.

## license and provenance

apache-2.0. this repo is almost entirely LLM-generated (as in, an AI wrote nearly all the code, the training scripts, and this readme).

training data comes from three public huggingface datasets, assembled by tools/build_dataset.py:

- [SupraLabs/chat-titles-filtered-115K](https://huggingface.co/datasets/SupraLabs/chat-titles-filtered-115K) (cc-by-4.0)
- [ogrnz/chat-titles](https://huggingface.co/datasets/ogrnz/chat-titles) (MIT)
- [Michionlion/chat-titles-english](https://huggingface.co/datasets/Michionlion/chat-titles-english) (cc-by-4.0)

the compare section uses the SupraLabs 50M title model
([supra-title-50M-pre-gguf](https://huggingface.co/SupraLabs/supra-title-50M-pre-gguf),
apache-2.0) as a reference only. it is not part of TinyTitle.
