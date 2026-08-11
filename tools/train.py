#!/usr/bin/env python3
"""Train the TinyTitle word-copy pointer-generator (english-only).

Architecture:
  - 8k unigram vocab (pad=0 eos=1 unk=2), shared emb [V,128]
  - 1-layer bidirectional GRU encoder, 128 per direction
  - 1-layer GRU decoder, 256
  - additive attention (256 -> 160 -> 1)
  - word-copy actions: generate_subword(token) | copy_source_word(group) | eos

Loss: NLL with label smoothing 0.05 on the vocab branch only, copy-gate
supervision (soft), coverage penalty, eos loss weight, masked to title tokens.

Usage:
    .venv/bin/python tools/train.py train --out models/tt-v1 --steps 20000 --bs 64 --lr 3e-3
    .venv/bin/python tools/train.py train --resume models/tt-v1/last.pt --steps 19750 --out models/tt-v1
    .venv/bin/python tools/train.py finetune --resume models/tt-v1/best.pt --out models/tt-v1-qat
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DEFAULT_TOK = os.path.join(PROC_DIR, "tok-8k.ttok")

D_EMB = 128
D_ENC = 128
D_DEC = 256
D_ATT = 160
MAX_SRC = 256
MAX_TGT = 16
MAX_BYTES = 6000
VOCAB_SIZE = 8000
COPY_BASE = VOCAB_SIZE

PAD, EOS, UNK = 0, 1, 2
LEXICAL_WORD_RE = re.compile(
    r"\.?(?:[A-Za-z0-9_\u00C0-\u024F]+(?:['’.\-][A-Za-z0-9_\u00C0-\u024F]+)*)(?:[+#]+)?")

STOPWORDS = set(
    "the a an and or but of to in on for with by from as at is are was were be been being "
    "it its this that these those i you he she we they me my your his her our their "
    "have has had do does did can could will would should may might not no yes about "
    "into over under what which who whom when where why how if then than so there here "
    "also very just get got one two three".split()
)

BENCH_RE = re.compile(
    r"(complete the sentence|fill in the blank|multiple choice|you will be given a definition"
    r"|sentence to rdf|translate the following|paraphrase the|rewrite the sentence|"
    r"step-by-step reasoning|which of the following|following question|following article"
    r"|read the following|answer the question|based on this review|make proper case|"
    r"summariz(e|ation))", re.I)
Q_PREFIX_RE = re.compile(r"^(q:|question:)", re.I)
TEACHER_RE = re.compile(r"^teacher:", re.I)


def load_vocab(tok_path):
    with open(tok_path, "rb") as f:
        assert f.read(5) == b"TTOK1"
        n = int.from_bytes(f.read(4), "little")
        vocab = []
        for _ in range(n):
            blen = f.read(1)[0]
            t = f.read(blen).decode("utf-8")
            idx = int.from_bytes(f.read(2), "little")
            vocab.append((idx, t, blen))
    vocab.sort()
    return [t for _, t, _ in vocab]


def tokenize_spans(text, tok):
    """Tokenize with byte spans, expanding unknown text to reserved byte ids."""
    enc = tok.encode(text)
    char_to_byte = [0]
    for ch in text:
        char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))

    ids, starts, ends = [], [], []
    for tid, (cs, ce) in zip(enc.ids, enc.offsets):
        bs, be = char_to_byte[cs], char_to_byte[ce]
        ids.append(tid)
        starts.append(bs)
        ends.append(be)
    return ids, starts, ends


# ---------------------------------------------------------------- selector

def sel_score(tokid, word, caps, d):
    s = 0.0
    if word in STOPWORDS:
        s -= 1.0
    if caps:
        s += 0.8
    if tokid == UNK:
        s += 1.5  # unknown source pieces are often identifier-like
    if re.search(r"\d", word):
        s += 0.5
    s += 0.5 * min(d, 2)
    return s


def select_source(ids, starts, ends, words, max_tokens=MAX_SRC):
    """Deterministic selection: keep first 192, salience-pick up to 64 later.

    Returns offs: list of indices into the original token arrays.
    """
    n = len(ids)
    if n <= max_tokens:
        return list(range(n))
    keep_first = 192
    head = list(range(keep_first))
    tail = list(range(keep_first, n))
    scored = []
    for i in tail:
        w = words[i] if i < len(words) else ""
        caps = bool(w) and w[0].isupper()
        s = sel_score(ids[i], w, caps, d=0)
        s += 0.3 * min(len(w), 8)
        s += 0.02 * (i - keep_first)
        scored.append((s, i))
    scored.sort(reverse=True)
    picked = [i for _, i in scored[:64]]
    picked.sort()
    return head + picked


# ---------------------------------------------------------------- data

def lexical_words(text):
    """Return lexical words as (surface, normalized, byte_start, byte_end)."""
    char_to_byte = [0]
    for ch in text:
        char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
    return [
        (m.group(0), m.group(0).casefold(), char_to_byte[m.start()], char_to_byte[m.end()])
        for m in LEXICAL_WORD_RE.finditer(text)
    ]


def build_dataset(path, tok_json, max_examples=None):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tok_json)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            msg, title = d["message"], d["title"]
            if len(msg.encode("utf-8")) > MAX_BYTES:
                continue
            if BENCH_RE.search(msg) or Q_PREFIX_RE.match(msg) or TEACHER_RE.match(msg):
                continue
            if len(msg) < 20:
                continue
            tw = title.split()
            if not (3 <= len(tw) <= 12):
                continue
            if not title[0].isalnum():
                continue
            m_ids, m_st, m_en = tokenize_spans(msg, t)
            if len(m_ids) > 4096:
                continue
            title_words = []
            generated_tokens = 0
            for surface, normalized, _, _ in lexical_words(title):
                word_ids = t.encode(surface).ids
                if not word_ids:
                    continue
                title_words.append((surface, normalized, word_ids))
                generated_tokens += len(word_ids)
            if not (2 <= len(title_words) <= 12) or generated_tokens > MAX_TGT - 1:
                continue
            rows.append((msg, title, m_ids, m_st, m_en, title_words))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def source_pieces(msg, starts, ends):
    raw = msg.encode("utf-8")
    return [raw[s:e].decode("utf-8", errors="ignore") for s, e in zip(starts, ends)]


class PGData:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msg, title, m_ids, m_st, m_en, title_words = self.rows[i]
        pieces = source_pieces(msg, m_st, m_en)
        offs = select_source(m_ids, m_st, m_en, pieces)
        return (offs, m_ids, m_st, m_en, pieces, title_words, msg, title)


def collate(items, max_src=MAX_SRC, max_tgt=MAX_TGT):
    # item: offs, token arrays, pieces, title_words, message, title
    src = [it[1] for it in items]
    st = [it[2] for it in items]
    en = [it[3] for it in items]
    B = len(items)
    L = max(1, min(max_src, max(len(it[0]) for it in items)))
    src_ids = torch.full((B, L), PAD, dtype=torch.long)
    src_st = torch.full((B, L), -1, dtype=torch.long)
    src_en = torch.full((B, L), -1, dtype=torch.long)
    src_offs = torch.full((B, L), -1, dtype=torch.long)
    word_ids = torch.full((B, L), -1, dtype=torch.long)
    action_rows = []

    for b, item in enumerate(items):
        offs, _, _, _, _, title_words, msg, _ = item
        sel = offs[:L]
        n = len(sel)
        src_ids[b, :n] = torch.tensor([src[b][i] for i in sel])
        src_st[b, :n] = torch.tensor([st[b][i] for i in sel])
        src_en[b, :n] = torch.tensor([en[b][i] for i in sel])
        src_offs[b, :n] = torch.tensor(sel)

        represented = {}
        groups = []
        source_words = lexical_words(msg)
        for j, original_idx in enumerate(sel):
            ts, te = st[b][original_idx], en[b][original_idx]
            for wi, (_, normalized, ws, we) in enumerate(source_words):
                if te > ws and ts < we:
                    if wi not in represented:
                        represented[wi] = len(groups)
                        groups.append((normalized, ws, we))
                    word_ids[b, j] = represented[wi]
                    break

        first_group = {}
        for group, (normalized, _, _) in enumerate(groups):
            first_group.setdefault(normalized, group)
        actions = []
        for _, normalized, token_ids in title_words:
            if normalized in first_group:
                actions.append(COPY_BASE + first_group[normalized])
            else:
                actions.extend(token_ids)
        actions.append(EOS)
        action_rows.append(actions[:max_tgt])

    T = max(1, max(len(row) for row in action_rows))
    tgt_actions = torch.full((B, T), PAD, dtype=torch.long)
    for b, actions in enumerate(action_rows):
        tgt_actions[b, :len(actions)] = torch.tensor(actions)
    return dict(src_ids=src_ids, src_st=src_st, src_en=src_en, src_offs=src_offs,
                word_ids=word_ids, tgt=tgt_actions, n_src=L, n_tgt=T)


# ---------------------------------------------------------------- model

class GRUCell(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.w = nn.Linear(in_dim, hid * 3, bias=False)
        self.uh = nn.Linear(hid, hid * 3, bias=False)
        self.b = nn.Parameter(torch.zeros(hid * 3))
        nn.init.orthogonal_(self.uh.weight)

    def forward(self, x, h):
        gx = self.w(x) + self.b
        gh = self.uh(h)
        xr, xz, xn = gx.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class BiGRUEncoder(nn.Module):
    def __init__(self, d_emb, d_enc):
        super().__init__()
        self.fwd = GRUCell(d_emb, d_enc)
        self.bwd = GRUCell(d_emb, d_enc)
        self.d_enc = d_enc

    def forward(self, x, lengths):
        B, L, D = x.shape
        hf = torch.zeros(B, self.d_enc, device=x.device, dtype=x.dtype)
        hb = torch.zeros(B, self.d_enc, device=x.device, dtype=x.dtype)
        out = torch.zeros(B, L, 2 * self.d_enc, device=x.device, dtype=x.dtype)
        for i in range(L):
            active_f = (i < lengths).unsqueeze(1)
            next_hf = self.fwd(x[:, i], hf)
            hf = torch.where(active_f, next_hf, hf)
            out[:, i, :self.d_enc] = torch.where(active_f, hf, torch.zeros_like(hf))

            active_b = (i < lengths).unsqueeze(1)
            j = (lengths - 1 - i).clamp_min(0)
            xj = x[torch.arange(B, device=x.device), j]
            next_hb = self.bwd(xj, hb)
            hb = torch.where(active_b, next_hb, hb)
            rows = torch.arange(B, device=x.device)[active_b.squeeze(1)]
            out[rows, j[rows], self.d_enc:] = hb[rows]
        return out


class PointerGenerator(nn.Module):
    def __init__(self, V, d_emb=D_EMB, d_enc=D_ENC, d_dec=D_DEC, d_att=D_ATT,
                 max_src=MAX_SRC, max_tgt=MAX_TGT, label_smooth=0.05, coverage_lambda=0.2,
                 eos_weight=1.0, copy_gate_weight=0.2):
        super().__init__()
        self.V = V
        self.d_emb = d_emb
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.d_att = d_att
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.label_smooth = label_smooth
        self.coverage_lambda = coverage_lambda
        self.eos_weight = eos_weight
        self.copy_gate_weight = copy_gate_weight
        self.emb = nn.Embedding(V, d_emb)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.enc = BiGRUEncoder(d_emb, d_enc)
        self.dec = GRUCell(d_emb, d_dec)
        self.dec_ctx = nn.Linear(2 * d_enc, d_emb, bias=False)
        self.w_att_v = nn.Linear(2 * d_enc, d_att, bias=False)
        self.w_att_h = nn.Linear(d_dec, d_att, bias=False)
        self.v_att = nn.Linear(d_att, 1, bias=False)
        self.cov = nn.Linear(1, d_att, bias=False)  # coverage -> attention space
        self.w_c = nn.Linear(2 * d_enc + d_dec, d_emb, bias=False)  # context -> emb
        self.w_o = nn.Linear(d_dec + d_emb, d_emb, bias=False)      # h + prev-emb
        self.gate = nn.Linear(d_dec + 2 * d_enc + d_emb, 1)         # copy gate
        self.head = nn.Linear(d_emb, V, bias=False)
        self.head.weight = self.emb.weight  # checkpoint compatibility; forward uses emb directly
        self.dec_init = nn.Linear(2 * d_enc, d_dec)

    def encode(self, src_ids):
        x = self.emb(src_ids)  # (B, L, d_emb)
        lengths = (src_ids != PAD).sum(dim=-1)
        return self.enc(x, lengths)

    def decode_step(self, dec_state, prev_emb, src_enc, pad_mask, cov_vec):
        B, L, _ = src_enc.shape
        h = dec_state
        e = self.w_att_v(src_enc) + self.w_att_h(h).unsqueeze(1)  # (B, L, d_att)
        if cov_vec is not None:
            e = e + self.cov(cov_vec.unsqueeze(-1))
        a = torch.tanh(e)
        scores = self.v_att(a).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * src_enc).sum(dim=1)  # (B, 2*d_enc)
        o = self.w_o(torch.cat([h, prev_emb], dim=-1))  # (B, d_emb)
        logit_in = self.w_c(torch.cat([h, ctx], dim=-1)) + o
        logits = F.linear(torch.tanh(logit_in), self.emb.weight)
        p_gen = torch.sigmoid(self.gate(torch.cat([h, ctx, prev_emb], dim=-1)))
        return logits, attn, ctx, p_gen

    def forward(self, batch):
        src_ids = batch["src_ids"]
        word_ids = batch["word_ids"]
        tgt = batch["tgt"]
        B, L = src_ids.shape
        T = tgt.shape[1]
        src_enc = self.encode(src_ids)  # (B, L, 2*d_enc)
        pad_mask = src_ids == PAD
        lengths = (src_ids != PAD).sum(dim=-1).clamp_min(1)
        rows = torch.arange(B, device=src_ids.device)
        summary = torch.cat([
            src_enc[rows, lengths - 1, :self.d_enc],
            src_enc[:, 0, self.d_enc:],
        ], dim=-1)
        h = torch.tanh(self.dec_init(summary))  # (B, d_dec)
        prev = torch.full((B,), PAD, dtype=torch.long, device=src_ids.device)
        prev_emb = self.emb(prev)
        cov = torch.zeros(B, L, device=src_ids.device)
        total_loss = torch.zeros((), device=src_ids.device)
        n_steps = 0
        for t in range(T):
            logits, attn, ctx, p_gen = self.decode_step(h, prev_emb, src_enc, pad_mask, cov)
            y = tgt[:, t]
            active = y != PAD
            is_copy = y >= self.V
            gen_target = y.masked_fill(is_copy | ~active, EOS)
            copy_target = (y - self.V).clamp(min=0, max=L - 1)

            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(1, gen_target.unsqueeze(1)).squeeze(1)
            smooth = -log_p.mean(dim=-1)
            word_prob = torch.zeros(B, L, device=attn.device, dtype=attn.dtype)
            valid_words = word_ids >= 0
            word_prob.scatter_add_(1, word_ids.clamp_min(0), attn * valid_words)
            copy_prob = word_prob.gather(1, copy_target.unsqueeze(1)).squeeze(1)

            gen = p_gen.squeeze(-1)
            action_prob = torch.where(is_copy, (1 - gen) * copy_prob, gen * torch.exp(-nll))
            step_loss = (1 - self.label_smooth) * -torch.log(action_prob.clamp_min(1e-9))
            step_loss = step_loss + self.label_smooth * gen * smooth * (~is_copy)
            step_loss = step_loss * torch.where(gen_target == EOS, self.eos_weight, 1.0)

            gate_target = (~is_copy).float().clamp(0.05, 0.95)
            gate_loss = F.binary_cross_entropy(gen, gate_target, reduction="none")
            coverage_loss = torch.minimum(attn, cov).sum(dim=-1)
            step_loss = step_loss + self.copy_gate_weight * gate_loss
            step_loss = step_loss + self.coverage_lambda * coverage_loss
            total_loss = total_loss + (step_loss * active).sum()
            n_steps += int(active.sum())
            cov = cov + attn.detach() * active.unsqueeze(1)

            generated_emb = self.emb(gen_target)
            group_mask = (word_ids == copy_target.unsqueeze(1)) & (src_ids != PAD)
            source_emb = self.emb(src_ids)
            copied_emb = (source_emb * group_mask.unsqueeze(-1)).sum(dim=1)
            copied_emb = copied_emb / group_mask.sum(dim=1, keepdim=True).clamp_min(1)
            prev_emb = torch.where(is_copy.unsqueeze(1), copied_emb, generated_emb)
            dec_input = prev_emb + torch.tanh(self.dec_ctx(ctx))
            next_h = self.dec(dec_input, h)
            h = torch.where(active.unsqueeze(1), next_h, h)
        return total_loss / max(1, n_steps)


# ---------------------------------------------------------------- int8 sim


class FakeQuantWeight(nn.Module):
    def forward(self, w):
        flat = w.reshape(w.shape[0], -1)
        scale = flat.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = (torch.clamp(torch.round(flat / scale), -127, 127) * scale).reshape_as(w)
        return w + (quantized - w).detach()


def quantize_per_row(w):
    """Per-row symmetric int8 with f32 scale. w: (out, in)."""
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.float()


def apply_int8(model):
    """Return a state dict with all Linear/Embedding weights replaced by int8 sim.

    Each entry becomes (scale_f32, q_int8) for weights; biases/LN stay float.
    """
    sd = {}
    for name, p in model.named_parameters():
        if "bias" in name or "cov" in name:
            sd[name] = p.detach().float()
            continue
        if name == "head.weight":
            sd[name] = ("tied",)
            continue
        q, s = quantize_per_row(p.detach().float())
        sd[name] = (s, q)
    return sd


def dequant(sd):
    out = {}
    for k, v in sd.items():
        if isinstance(v, tuple):
            if v[0] == "tied":
                continue
            s, q = v
            out[k] = (q.float() * s).float()
        else:
            out[k] = v
    out["head.weight"] = out["emb.weight"]
    return out


# ---------------------------------------------------------------- training

def run_train(args):
    tok_json = os.path.join(PROC_DIR, "tokenizer-8k.json")
    if not os.path.exists(tok_json) or not os.path.exists(DEFAULT_TOK):
        raise FileNotFoundError(
            "missing tokenizer artifacts; run tools/train_tokenizer.py so json and ttok stay paired")

    train_rows = build_dataset(os.path.join(PROC_DIR, "train.jsonl"), tok_json)
    dev_rows = build_dataset(os.path.join(PROC_DIR, "dev.jsonl"), tok_json)
    vocab = load_vocab(DEFAULT_TOK)
    print(f"train {len(train_rows)} dev {len(dev_rows)} vocab {len(vocab)}")

    model = PointerGenerator(len(vocab))
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        if not isinstance(ck, dict) or ck.get("objective") != "word-copy":
            raise ValueError("incompatible checkpoint: expected word-copy; train from scratch")
        model.load_state_dict(ck["model"])
        print(f"resumed {args.resume}")
    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # bucket by source length
    rng = random.Random(args.seed)
    buckets = {}
    for i, r in enumerate(train_rows):
        b = min(len(r[2]) // 32, 12)
        buckets.setdefault(b, []).append(i)
    order = []
    for b in sorted(buckets):
        rng.shuffle(buckets[b])
        order.extend(buckets[b])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    step = 0
    total_steps = args.steps

    if args.resume:
        step = int(ck.get("step", 0))
        total_steps = step + args.steps
        if isinstance(ck, dict) and "opt" in ck:
            opt.load_state_dict(ck["opt"])

        print(f"resumed {args.resume} at step {step}; training to {total_steps}", flush=True)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=total_steps)
    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    start_step = step
    run_steps = total_steps - start_step
    t_start = time.perf_counter()
    report_time = t_start
    report_completed = 0
    report_loss = 0.0
    report_count = 0
    bar_active = False
    validation_status = ""
    last_progress_line = ""

    def make_batch(rows, idxs):
        items = []
        for j in idxs:
            msg, title, m_ids, m_st, m_en, title_words = rows[j]
            pieces = source_pieces(msg, m_st, m_en)
            offs = select_source(m_ids, m_st, m_en, pieces)
            items.append((offs, m_ids, m_st, m_en, pieces, title_words, msg, title))
        batch = collate(items)
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    bar_len = 32
    interactive = sys.stdout.isatty()

    def show_progress(force=False):
        nonlocal report_time, report_completed, report_loss, report_count
        nonlocal bar_active, last_progress_line
        completed = step - start_step
        if not force and completed % 50 != 0:
            return
        now = time.perf_counter()
        delta_steps = completed - report_completed
        rate = delta_steps / max(1e-9, now - report_time)
        avg_loss = report_loss / max(1, report_count)
        pct = completed / max(1, run_steps)
        filled = min(bar_len, int(bar_len * pct))
        eta = (run_steps - completed) / rate if rate > 0 else 0.0
        line = (f"[{'#' * filled}{'-' * (bar_len - filled)}] {pct * 100:6.2f}% "
                f"run {completed}/{run_steps} global {step}/{total_steps} "
                f"loss {avg_loss:.4f} {rate:.2f} it/s eta {eta / 60:.1f}m")
        last_progress_line = line
        suffix = f" | {validation_status}" if validation_status else ""
        prefix = "\r\033[2K" if interactive else ""
        print(prefix + line + suffix, end="" if interactive else "\n", flush=True)
        bar_active = interactive
        report_time = now
        report_completed = completed
        report_loss = 0.0
        report_count = 0

    while step < total_steps:
        rng.shuffle(order)
        for i in range(0, len(order), args.bs):
            if step >= total_steps:
                break
            idxs = order[i:i + args.bs]
            batch = make_batch(train_rows, idxs)
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            report_loss += loss.item()
            report_count += 1
            show_progress(force=step == total_steps)
            if step % 250 == 0:
                model.eval()
                with torch.no_grad():
                    dev_loss = 0.0
                    n = 0
                    for k in range(0, len(dev_rows), 32):
                        idxs = list(range(k, min(k + 32, len(dev_rows))))
                        b = make_batch(dev_rows, idxs)
                        dev_loss += model(b).item() * len(idxs)
                        n += len(idxs)
                    dev_loss /= max(1, n)
                    if dev_loss < best:
                        best = dev_loss
                        torch.save({"objective": "word-copy", "model": model.state_dict(),
                                    "loss": dev_loss, "step": step},
                                   os.path.join(args.out, "best.pt"))
                        validation_status = f"dev {dev_loss:.4f} best saved"
                    else:
                        validation_status = f"dev {dev_loss:.4f} best {best:.4f}"
                    if interactive:
                        print(f"\r\033[2K{last_progress_line} | {validation_status}", end="", flush=True)
                        bar_active = True
                    else:
                        print(f"step {step} {validation_status}", flush=True)
                    # periodic checkpoint for resume
                    torch.save({"objective": "word-copy", "model": model.state_dict(),
                                "loss": loss.item(), "step": step, "opt": opt.state_dict(),
                                "sched": sched.state_dict()},
                               os.path.join(args.out, "last.pt"))
                model.train()
                # evaluation time must not pollute the training throughput/eta.
                report_time = time.perf_counter()
                report_completed = step - start_step
    if bar_active:
        print(flush=True)
    print(f"done in {(time.perf_counter()-t_start)/60:.1f}m. best dev loss {best:.4f} -> {os.path.join(args.out, 'best.pt')}")


def run_finetune(args):
    """Short int8 QAT fine-tune starting from a fp32 checkpoint."""
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    tok_json = os.path.join(PROC_DIR, "tokenizer-8k.json")
    train_rows = build_dataset(os.path.join(PROC_DIR, "train.jsonl"), tok_json)
    vocab = load_vocab(DEFAULT_TOK)
    model = PointerGenerator(len(vocab))
    sd = torch.load(args.resume, map_location="cpu")
    if not isinstance(sd, dict) or sd.get("objective") != "word-copy":
        raise ValueError("incompatible checkpoint: expected word-copy")
    model.load_state_dict(sd["model"])
    print(f"loaded {args.resume}")

    # apply differentiable fake quantization to every runtime-quantized weight.
    from torch.nn.utils import parametrize
    quantized_modules = []
    for module_name, module in model.named_modules():
        if module is model.head:
            continue
        if isinstance(module, (nn.Linear, nn.Embedding)) and hasattr(module, "weight"):
            parametrize.register_parametrization(module, "weight", FakeQuantWeight())
            quantized_modules.append(module)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    rng = random.Random(args.seed)
    interactive = sys.stdout.isatty()
    bar_len = 32
    t_start = time.perf_counter()
    report_time = t_start
    report_step = 0
    report_loss = 0.0
    report_count = 0
    for step in range(args.steps):
        items = []
        for j in rng.sample(range(len(train_rows)), min(args.bs, len(train_rows))):
            msg, title, m_ids, m_st, m_en, title_words = train_rows[j]
            pieces = source_pieces(msg, m_st, m_en)
            offs = select_source(m_ids, m_st, m_en, pieces)
            items.append((offs, m_ids, m_st, m_en, pieces, title_words, msg, title))
        batch = collate(items)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        loss = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        completed = step + 1
        report_loss += loss.item()
        report_count += 1
        if completed % 50 == 0 or completed == args.steps:
            now = time.perf_counter()
            rate = (completed - report_step) / max(1e-9, now - report_time)
            pct = completed / max(1, args.steps)
            filled = min(bar_len, int(bar_len * pct))
            eta = (args.steps - completed) / rate if rate > 0 else 0.0
            line = (f"[{'#' * filled}{'-' * (bar_len - filled)}] {pct * 100:6.2f}% "
                    f"qat {completed}/{args.steps} loss {report_loss / report_count:.4f} "
                    f"{rate:.2f} it/s eta {eta / 60:.1f}m")
            print(("\r" if interactive else "") + line,
                  end="" if interactive and completed < args.steps else "\n", flush=True)
            report_time = now
            report_step = completed
            report_loss = 0.0
            report_count = 0
    for module in quantized_modules:
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    model.head.weight = model.emb.weight
    os.makedirs(args.out, exist_ok=True)
    torch.save({"objective": "word-copy", "model": model.state_dict(),
                "loss": loss.item(), "qat": True},
               os.path.join(args.out, "best.pt"))
    print(f"qat done -> {os.path.join(args.out, 'best.pt')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join("models", "pg"))
    p.add_argument("--resume", default=None)
    p.set_defaults(fn=run_train)
    p = sub.add_parser("finetune")
    p.add_argument("--resume", required=True)
    p.add_argument("--out", default=os.path.join("models", "pg-qat"))
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=run_finetune)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
