# TinyTitle (design)

> status: implemented and smoke-tested, english-only. this document is the source of truth for the current word-copy training objective and the `.ttm1` c runtime contract.

## 1. product contract

input:

- one english chat message or conversation prefix.
- at most 6,000 utf-8 bytes.
- malformed utf-8 and non-whitespace control bytes are rejected.
- neural work is bounded to at most 256 selected tokenizer tokens.

output:

- one short title on stdout.
- titles are normally 2 to 10 rendered words.
- generation is deterministic by default.
- the runtime is a one-shot, single-threaded c process with no external inference dependencies.

hard resource target:

- peak total rss below 10 mib, including the process, mapped model, tokenizer, activations, and decoder state.
- low cpu usage is preferred over throughput or general language-model capability.

smoke measurements on the project bench machine:

| item | measured value |
|---|---:|
| parameters | 1,813,185 |
| int8 artifact | 1,980,559 bytes |
| peak rss | 4.89 mib |
| median benchmark wall time | about 25 ms |
| long 6,000-byte benchmark | about 122 ms |
| tokenizer parity | 9/9 fixtures |
| c/python decode parity | 30/30 smoke cases |

## 2. high-level pipeline

```text
utf-8 message
  -> metaspace unigram tokenization with byte spans
  -> deterministic 256-token source selection
  -> lexical source-word grouping
  -> bidirectional gru encoder
  -> attentive gru decoder
  -> disjoint action distribution:
       generate_subword(token)
       copy_source_word(word_index)
       eos
  -> greedy decode by default, optional beam-2
  -> duplicate-word cleanup
  -> title
```

this is a task-specific sequence transducer, not a general language model.

## 3. tokenizer

artifacts:

```text
data/processed/tokenizer-8k.json
data/processed/tok-8k.ttok
```

configuration:

- english-trained unigram tokenizer.
- vocabulary size: 8,000.
- metaspace boundary marker: `▁`.
- special ids:
  - `0`: `<pad>`
  - `1`: `<eos>`
  - `2`: `<unk>`
- tokenizer training uses only `train.jsonl`; dev and test data are not included.
- unknown unicode fallback consumes one complete utf-8 code point, never an individual byte.

metaspace preserves word boundaries inside token strings:

```text
"network setup" -> ["▁network", "▁set", "up"]
```

both python and c perform the same unigram viterbi segmentation using scores embedded in the model artifact.

for every source token, tokenization also records byte offsets into the original input. copied output can therefore preserve original casing and spelling.

## 4. bounded source selection

messages may contain far more than 256 tokenizer tokens, but the neural source is fixed at 256.

selection policy:

1. keep the first 192 tokens.
2. score all remaining tokens.
3. select the highest-scoring 64 tail tokens.
4. restore selected tail tokens to source order.

salience rewards:

- capitalization.
- digits.
- unknown/identifier-like pieces.
- longer token spans.
- later tail position, slightly.

common stopwords are penalized in python training selection. selected tokens retain their original byte offsets.

this bound controls encoder cpu, attention workspace, and recurrent-state memory. there is no allocation proportional to more than the accepted 6,000-byte input or fixed tokenizer limits.

## 5. lexical source words

the model groups selected source tokens into lexical words. a lexical word is not the same as a whitespace-delimited field.

supported forms include:

```text
network
résumé
nielsen's
node.js
gpt-4
.net
c++
c#
```

surrounding sentence punctuation is excluded:

```text
cost?       -> copy span "cost"
universe,   -> copy span "universe"
question:   -> copy span "question"
```

current lexical character contract:

- ascii letters and digits.
- underscore.
- latin unicode letters from `u+00c0` through `u+024f`.
- internal apostrophe, curly apostrophe, period, and hyphen when followed by another lexical character.
- optional leading period for forms such as `.net`.
- optional trailing `+` or `#` for forms such as `c++` and `c#`.

python training, the python reference, and the c runtime implement the same contract.

only lexical words represented by at least one selected source token receive copy actions. a selected token has either one lexical word id or `-1` when it represents punctuation/non-lexical text.

## 6. training action construction

the target is not a flat title subword sequence.

for each lexical word in a gold title:

1. normalize it with unicode-aware case folding.
2. search represented source lexical words for the same normalized form.
3. if found, append one `copy_source_word(group)` action.
4. otherwise tokenize the title word and append its generated subword ids.
5. append generated `<eos>` after the title.

runtime action ids are conceptual rather than stored in the model vocabulary:

```text
0 .. 7999             generated vocabulary actions
8000 + source_group   copied source-word actions
```

copy action ids exist only in a training batch. the number of source groups is dynamic and bounded by 256.

example:

```text
source: "Why does diesel fuel cost more today?"
gold:   "Diesel fuel price increase"

actions:
  copy_word("diesel")
  copy_word("fuel")
  generate("▁price")
  generate("▁increase")
  eos
```

punctuation from `today?` cannot enter a copied title word because `?` is outside the lexical span.

## 7. model architecture

| component | configuration |
|---|---:|
| vocabulary | 8,000 |
| embedding dimension | 128 |
| encoder | one bidirectional gru layer |
| encoder hidden | 128 per direction |
| encoder output | 256 |
| decoder | one gru layer |
| decoder hidden | 256 |
| additive attention | 160 |
| maximum source | 256 tokenizer tokens |
| maximum target actions | 16 |
| unique parameters | 1,813,185 |

### encoder

input token embeddings are shared with the generated-token output projection.

```text
embedding: [vocab, 128]
forward gru: 128 hidden
backward gru: 128 hidden
source encoding: concat(forward, backward) -> 256
```

padding does not update recurrent state. backward states are written to their corresponding source positions.

### decoder initialization

the initial decoder state uses both final encoder directions:

```text
summary = concat(
  final forward state at the last real token,
  final backward state at the first token
)

decoder_state = tanh(dec_init(summary))
```

### attention

attention is additive attention with coverage:

```text
e_i = w_source(source_i) + w_state(decoder_state) + w_coverage(coverage_i)
score_i = v(tanh(e_i))
attention = softmax(score)
context = sum(attention_i * source_i)
```

padded source positions are masked in training. runtime sources contain no padded positions.

### generated vocabulary branch

```text
output_hidden = tanh(
  w_context([decoder_state; context])
  + w_output([decoder_state; previous_embedding])
)

vocab_logits = shared_embedding_matrix * output_hidden
p_vocab = softmax(vocab_logits)
```

### action gate

```text
p_generate = sigmoid(gate([decoder_state; context; previous_embedding]))
```

the final action distribution is disjoint:

```text
p(generate token v) = p_generate * p_vocab(v)

p(copy word g) =
  (1 - p_generate)
  * sum(attention_i for source tokens whose word_id == g)
```

a generated token and copied word do not compete through the same token id. copying is explicitly word-indexed.

## 8. training loss

for a generated action:

```text
loss = -log(p_generate * p_vocab(gold_token))
```

for a copied-word action:

```text
loss = -log((1 - p_generate) * word_attention(gold_word_group))
```

additional terms:

- label smoothing `0.05` on generated vocabulary actions only.
- gate binary cross-entropy:
  - generated action target: approximately `0.95`.
  - copied action target: approximately `0.05`.
- coverage penalty using `min(current_attention, previous_coverage)`.
- configurable eos weight.
- padding actions are excluded from all loss terms and recurrent updates.

default loss weights:

```text
coverage lambda: 0.2
copy gate weight: 0.2
eos weight: 1.0
```

### teacher-forced decoder feedback

for a generated token:

```text
previous_embedding = embedding(gold_token)
```

for a copied word:

```text
previous_embedding = mean(
  embeddings of selected source tokens belonging to the copied word
)
```

source context also enters the decoder update:

```text
decoder_input = previous_embedding + tanh(decoder_context_projection(context))
decoder_state = gru(decoder_input, decoder_state)
```

this exact feedback rule is implemented in python and c.

## 9. inference actions

for each hypothesis and decoder step:

1. run attention and compute `p_generate`.
2. compute generated token probabilities over all 8,000 vocabulary ids.
3. sum token attention into lexical source-word probabilities.
4. rank generated-token and copied-word actions together.
5. apply the selected action.
6. feed the selected action embedding and attended context into the decoder gru.
7. add attention to coverage.

### generated action rendering

- decode the vocabulary token.
- replace metaspace `▁` with a normal space.
- append it to the title buffer.

### copied-word rendering

- append one separating space when needed.
- copy the complete lexical byte span from the original input.
- do not include surrounding sentence punctuation.
- preserve original casing, accents, dots, apostrophes, hyphens, `+`, and `#` inside the lexical word.

there is no partial-word rewind, copy lock, or post-hoc subword expansion.

## 10. decoding modes

### default: greedy

greedy decoding is the production default because it minimizes cpu use.

```bash
runtime/title-v1 models/tt-v1/model.ttm1 "message"
```

### optional beam-2

beam width two is implemented as a bounded runtime option:

```bash
TTM_BEAM_WIDTH=2 runtime/title-v1 models/tt-v1/model.ttm1 "message"
```

each beam hypothesis owns fixed-size:

- decoder state.
- previous action embedding.
- 256-position coverage vector.
- score and action count.
- 512-byte title buffer.

ranking uses a `0.6` length penalty. eos is suppressed until at least two rendered words exist. beam-2 can modestly improve automatic quality but approximately doubles decoder cpu, so it is opt-in.

## 11. output cleanup

final cleanup removes duplicate rendered words globally and case-insensitively. this prevents title forms such as:

```text
human universe unique in universe
```

cleanup does not invent, reorder, stem, or otherwise rewrite words.

because title training uses lexical words and copied spans exclude sentence punctuation, punctuation should not normally appear in generated titles. technical punctuation inside recognized lexical identifiers is preserved.

## 12. quantization

shipped weights use per-row symmetric int8:

```text
scale = max(abs(row)) / 127
q = clamp(round(row / scale), -127, 127)
```

runtime matrix-vector multiplication accumulates directly from int8 rows into float32 and applies the row scale once. it does not materialize a dequantized matrix or dequantized row for normal matvec operations.

kept as float32:

- one-dimensional biases.
- quantization scales.

post-training int8 is the default. optional qat exists but should only be used if a measured fp32/int8 quality gap justifies it.

## 13. model format and compatibility

file extension: `.ttm1`

internal magic: `TTM1`

the magic is the format version; future incompatible formats increment it (`TTM2`, ...). the word-copy model family is identified by the checkpoint `objective` field, not by the file format.

sections:

1. vocabulary strings and unigram scores.
2. named int8/f32 tensors.

checkpoint metadata:

```text
objective = "word-copy"
```

safety rules:

- trainer resume rejects checkpoints without `word-copy`.
- qat rejects incompatible checkpoints.
- exporter rejects incompatible checkpoints.
- runtime rejects bad magic and incompatible dimensions.

## 14. c runtime memory layout

weights and tokenizer are mapped read-only with `mmap`.

bounded allocations include:

- selected token arrays: at most 256.
- lexical source groups: at most 256.
- encoder outputs: `256 x 256` float32 maximum.
- attention and coverage: 256 float32 entries per hypothesis.
- activation scratch sized from fixed dimensions.
- generated logits/probabilities: 8,000 float32 entries each.
- at most two beam hypotheses.
- 512-byte output buffer per beam.

there is:

- no transformer kv cache.
- no graph allocator.
- no allocation inside the decode loop.
- no blas, openmp, python, pytorch, or llama.cpp dependency in the shipped runtime.
- no background process or idle cpu use.

all-pages-touched rss is measured with `TTM_TOUCH=1`.

## 15. file map

```text
tools/train_tokenizer.py
  trains the paired 8k metaspace unigram tokenizer artifacts

tools/train.py
  builds lexical word-copy actions and trains the attentive gru

runtime/export.py
  validates word-copy checkpoints and exports ttm1/int8

runtime/ttm.h
  validated mmap loader, tokenizer, vocabulary, and int8 operations

runtime/main.c
  source selection, lexical grouping, encoder, action decoder, beam, rendering

tools/ref.py
  numpy reference implementing the same tokenizer, model, word actions, and beam

tools/parity.py
  tokenizer and c/python deterministic decode parity

tools/eval.py
  rouge-l, chr-f, token-f1, output diagnostics, and latency

tools/bench.py
  bounded-input rss and cpu benchmarks

tools/gate.py
  combined quality/resource report; defaults to runtime/title-v1

tools/compare.py
  side-by-side title comparison vs the SupraLabs 50M reference gguf

tools/build_dataset.py
  assembles train/dev/test jsonl from the public chat-title datasets

tools/build_manifest.py
  builds the fixed stratified human-gate manifest
```

## 16. training and release workflow

current checkpoints:

```text
models/tt-v1/last.pt           full 20k-step word-copy model
```

export:

```bash
.venv/bin/python runtime/export.py \
  --ckpt models/tt-v1/best.pt \
  --out models/tt-v1/model.ttm1
```

build:

```bash
cc -std=c11 -O3 -march=native -DNDEBUG \
  -o runtime/title-v1 runtime/main.c -lm
```

parity:

```bash
.venv/bin/python tools/parity.py \
  --ttm1 models/tt-v1/model.ttm1 \
  --runtime runtime/title-v1 \
  --limit 100
```

full gate:

```bash
.venv/bin/python tools/gate.py \
  --ttm1 models/tt-v1/model.ttm1 \
  --runtime runtime/title-v1 \
  --limit 2000
```

manual inference:

```bash
runtime/title-v1 models/tt-v1/model.ttm1 \
  "Every few minutes my laptop loses internet access, but every other device remains connected."
```

## 17. correctness invariants

changes are not release-ready unless all applicable invariants hold:

- training and runtime use the same tokenizer ids.
- unknown unicode fallback consumes complete code points.
- source lexical spans agree between python and c.
- sentence punctuation is not part of copied words.
- every copied-word probability equals the sum of token attention assigned to that word.
- copied-word teacher feedback equals the mean source-token embedding for that word.
- generated and copied actions are disjoint.
- c and python greedy output match on parity fixtures.
- beam state is independent per hypothesis.
- checkpoints without the `word-copy` objective and files without `TTM1` magic are rejected.
- model pages are pre-touched for rss acceptance.
- worst-case rss remains below 10 mib.

## 18. known limitations

- copying is clean, but abstraction is still limited by the small supervised model and dataset.
- copied words must match title words after case folding during training; synonyms require generation.
- the deterministic source selector can omit a useful late source word.
- only ascii and latin-range letters participate in lexical copy words; this is deliberate for english-only v1.
- generated subwords can theoretically form an unusual word, although they can no longer splice with a copied word action.
- automatic overlap metrics do not fully measure fluency or title usefulness.
- beam-2 costs substantially more cpu and is not the default.
- the test data contains benchmark-like prompts that differ from ordinary chat-title traffic; report both aggregate metrics and qualitative chat examples.
