#!/usr/bin/env python3
"""Python reference for the TinyTitle pointer-generator runtime (.ttm1).

Implements the same metaspace unigram tokenizer, selector, corrected gru,
attention, word-copy actions, and greedy/beam decode as runtime/main.c.
"""
import os
import re
import struct
import sys

import numpy as np

D_EMB, D_ENC_HALF, D_ENC, D_DEC, D_ATT = 128, 128, 256, 256, 160
MAX_SRC = 256
MAX_TGT = 16
LEXICAL_WORD_RE = re.compile(
    r"\.?(?:[A-Za-z0-9_\u00C0-\u024F]+(?:['’.\-][A-Za-z0-9_\u00C0-\u024F]+)*)(?:[+#]+)?")


def lexical_spans(text):
    char_to_byte = [0]
    for ch in text:
        char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
    return [(char_to_byte[m.start()], char_to_byte[m.end()])
            for m in LEXICAL_WORD_RE.finditer(text)]


class TTM1:
    def __init__(self, path):
        data = open(path, "rb").read()
        self.data = data
        assert data[:4] == b"TTM1"
        nv = struct.unpack("<I", data[4:8])[0]
        ns = struct.unpack("<I", data[8:12])[0]
        tbl = struct.unpack("<Q", data[12:20])[0]
        self.vocab_size = nv
        self.sections = {}
        for i in range(ns):
            tag, pad, off, sz = struct.unpack("<IIQQ", data[tbl + i * 24: tbl + i * 24 + 24])
            self.sections[tag] = (off, sz)
        self._parse_vocab()
        self._parse_weights()

    def _parse_vocab(self):
        off, sz = self.sections[1]
        p = off + 4
        count = struct.unpack("<I", self.data[off:off + 4])[0]
        self.vocab = []
        self.scores = []
        for _ in range(count):
            bl = self.data[p]
            s = self.data[p + 1:p + 1 + bl].decode("utf-8")
            score = struct.unpack("<f", self.data[p + 1 + bl + 2: p + 1 + bl + 6])[0]
            self.vocab.append(s)
            self.scores.append(score)
            p += 1 + bl + 6

    def _parse_weights(self):
        off, sz = self.sections[2]
        p = off + 4
        count = struct.unpack("<I", self.data[off:off + 4])[0]
        self.tensors = {}
        for _ in range(count):
            nl = struct.unpack("<I", self.data[p:p + 4])[0]
            p += 4
            name = self.data[p:p + nl].decode()
            p += nl
            p += (4 - (p % 4)) % 4  # name padded to 4-byte alignment
            dims = struct.unpack("<I", self.data[p:p + 4])[0]
            p += 4
            a, b = struct.unpack("<ii", self.data[p:p + 8])
            p += 8
            if dims == 2:
                rows, row_bytes = struct.unpack("<QQ", self.data[p:p + 16])
                p += 16
                if row_bytes == 0:
                    # raw f32
                    self.tensors[name] = np.frombuffer(self.data[p:p + 4 * a * b], dtype="<f4").astype(np.float32).reshape(a, b)
                    p += 4 * a * b
                    continue
                scales = np.frombuffer(self.data[p:p + 4 * rows], dtype="<f4").astype(np.float32)
                p += 4 * rows
                q = np.frombuffer(self.data[p:p + row_bytes], dtype="<i1").astype(np.float32).reshape(rows, b)
                p += row_bytes
                self.tensors[name] = scales[:, None] * q
            else:
                self.tensors[name] = np.frombuffer(self.data[p:p + 4 * a], dtype="<f4").astype(np.float32)
                p += 4 * a
        self.tensors["head.weight"] = self.tensors["emb.weight"]

    # ---- tokenizer: metaspace unigram over whitespace pieces ----
    def tokenize(self, text):
        raw = text.encode("utf-8")
        ids, starts, ends = [], [], []
        i = 0
        while i < len(raw):
            while i < len(raw) and raw[i] in b" \t\n\r\v\f":
                i += 1
            if i >= len(raw):
                break
            j = i
            while j < len(raw) and raw[j] not in b" \t\n\r\v\f":
                j += 1
            pids, pst, pen = self._viterbi("▁".encode() + raw[i:j])
            for tid, ps, pe in zip(pids, pst, pen):
                ids.append(tid)
                starts.append(i + max(0, min(j - i, ps - 3)))
                ends.append(i + max(0, min(j - i, pe - 3)))
            i = j
        return ids, starts, ends

    def _viterbi(self, buf):
        n = len(buf)
        if n == 0:
            return [], [], []
        NEG = -1e18
        dp = [NEG] * (n + 1)
        prev = [-1] * (n + 1)
        tok = [0] * (n + 1)
        dp[0] = 0
        # token trie for speed
        for i in range(n):
            if dp[i] == NEG:
                continue
            for v, s in enumerate(self.vocab):
                b = s.encode("utf-8")
                if i + len(b) <= n and buf[i:i + len(b)] == b:
                    sc = dp[i] + self.scores[v]
                    if sc > dp[i + len(b)]:
                        dp[i + len(b)] = sc
                        prev[i + len(b)] = i
                        tok[i + len(b)] = v
            # unknown fallback consumes one complete utf-8 code point.
            lead = buf[i]
            cp = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
            cp = min(cp, n - i)
            sc = dp[i] - 20.0
            if sc > dp[i + cp]:
                dp[i + cp] = sc
                prev[i + cp] = i
                tok[i + cp] = 2
        ids, st, en = [], [], []
        pos = n
        while pos > 0:
            p0 = prev[pos]
            ids.append(tok[pos])
            st.append(p0)
            en.append(pos)
            pos = p0
        return ids[::-1], st[::-1], en[::-1]

    # ---- selector (matches runtime) ----
    def select(self, ids, starts, ends, text):
        n = len(ids)
        if n <= MAX_SRC:
            return list(range(n))
        keep = 192
        scored = []
        for i in range(keep, n):
            raw = text.encode("utf-8")
            word = raw[starts[i]:ends[i]]
            s = 0.0
            if word and b"A" <= word[:1] <= b"Z":
                s += 0.8
            if any(ord("0") <= c <= ord("9") for c in word):
                s += 0.5
            if ids[i] == 2:
                s += 1.5
            s += 0.3 * min(len(word), 8)
            s += 0.02 * (i - keep)
            scored.append((s, i))
        scored.sort(reverse=True)
        picked = [i for _, i in scored[:64]]
        picked.sort()
        return list(range(keep)) + picked

    # ---- ops ----
    @staticmethod
    def _sigmoid(x):
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        positive = x >= 0
        out[positive] = 1 / (1 + np.exp(-x[positive]))
        ex = np.exp(x[~positive])
        out[~positive] = ex / (1 + ex)
        return out

    def _gru(self, name, x, h):
        w = self.tensors[name + ".w.weight"]
        u = self.tensors[name + ".uh.weight"]
        b = self.tensors[name + ".b"]
        gx = x @ w.T + b
        gh = h @ u.T
        hid = gx.shape[-1] // 3
        r = self._sigmoid(gx[..., :hid] + gh[..., :hid])
        z = self._sigmoid(gx[..., hid:2 * hid] + gh[..., hid:2 * hid])
        n = np.tanh(gx[..., 2 * hid:] + r * gh[..., 2 * hid:])
        return (1 - z) * n + z * h

    def encode(self, ids):
        x = self.tensors["emb.weight"][ids]
        n = len(ids)
        hf = np.zeros(D_ENC_HALF, np.float32)
        hb = np.zeros(D_ENC_HALF, np.float32)
        fwd = np.zeros((n, D_ENC_HALF), np.float32)
        bwd = np.zeros((n, D_ENC_HALF), np.float32)
        for i in range(n):
            hf = self._gru("enc.fwd", x[i], hf)
            fwd[i] = hf
        for i in range(n - 1, -1, -1):
            hb = self._gru("enc.bwd", x[i], hb)
            bwd[i] = hb
        return np.concatenate([fwd, bwd], axis=1)

    def decode_step(self, h, prev_emb, src_enc, cov, n):
        # attention scores
        wh = self.tensors["w_att_h.weight"] @ h  # (128,)
        e = src_enc @ self.tensors["w_att_v.weight"].T + wh  # (n, 128)
        # cov.weight is [128, 1]; per-position cov[i] adds cov_weight[:,0] * cov[i]
        e = e + self.tensors["cov.weight"][:, 0] * cov[:, None]  # (n, 128)
        scores = (np.tanh(e) @ self.tensors["v_att.weight"].T)[:, 0]  # (n,)
        mx = scores.max()
        attn = np.exp(scores - mx)
        attn = attn / attn.sum()
        ctx = attn @ src_enc  # (192,)
        # output
        li = self.tensors["w_c.weight"] @ np.concatenate([h, ctx]) + self.tensors["w_o.weight"] @ np.concatenate([h, prev_emb])
        li = np.tanh(li)
        logits = self.tensors["emb.weight"] @ li  # (V,)
        # copy gate
        g = (self.tensors["gate.weight"] @ np.concatenate([h, ctx, prev_emb]) +
             self.tensors["gate.bias"])
        p_gen = self._sigmoid(g)
        return logits, attn, p_gen, ctx

    def decode(self, text, max_tokens=16, temp=1.0):
        ids, starts, ends = self.tokenize(text)
        offs = self.select(ids, starts, ends, text)
        sel_ids = [ids[i] for i in offs]
        src_enc = self.encode(sel_ids)
        summary = np.concatenate([src_enc[-1, :D_ENC_HALF], src_enc[0, D_ENC_HALF:]])
        initial_h = np.tanh(self.tensors["dec_init.weight"] @ summary + self.tensors["dec_init.bias"])
        raw = text.encode("utf-8")
        source_words = lexical_spans(text)
        represented = {}
        token_word = []
        words = []
        for original_idx in offs:
            ts, te = starts[original_idx], ends[original_idx]
            group = -1
            for wi, (ws, we) in enumerate(source_words):
                if te > ws and ts < we:
                    if wi not in represented:
                        represented[wi] = len(words)
                        words.append((ws, we))
                    group = represented[wi]
                    break
            token_word.append(group)
        V = self.vocab_size

        def rank(hyp):
            if hyp["done"] and len(hyp["title"].split()) < 2:
                return -1e30
            penalty = ((5 + max(1, hyp["tokens"])) / 6) ** 0.6
            return hyp["score"] / penalty

        def append_generated(hyp, token):
            hyp["title"] += self.vocab[token].replace("▁", " ")

        def append_copied(hyp, group):
            bs, be = words[group]
            word = raw[bs:be].decode("utf-8", "strict")
            hyp["title"] = (hyp["title"].rstrip() + " " + word).strip()

        beam_width = 2 if os.environ.get("TTM_BEAM_WIDTH") == "2" else 1
        beams = [{
            "h": initial_h,
            "prev": self.tensors["emb.weight"][0],
            "cov": np.zeros(len(sel_ids), np.float32),
            "score": 0.0, "tokens": 0, "done": False, "title": "",
        }]
        for t in range(max_tokens):
            candidates = []
            for parent in beams:
                if parent["done"]:
                    candidates.append(parent)
                    continue
                logits, attn, p_gen_arr, ctx = self.decode_step(
                    parent["h"], parent["prev"], src_enc, parent["cov"], len(sel_ids))
                p_gen = float(p_gen_arr[0])
                pv = np.exp((logits - logits.max()) / temp)
                pv /= pv.sum()
                probs = p_gen * pv
                if t < 2 or len(parent["title"].split()) < 2:
                    probs[1] = 0
                probs[0] = 0
                word_probs = np.zeros(len(words), np.float32)
                for i, group in enumerate(token_word):
                    if group >= 0:
                        word_probs[group] += (1 - p_gen) * attn[i]
                top = [(float(prob), action) for action, prob in enumerate(probs) if prob > 0]
                top.extend((float(prob), V + group) for group, prob in enumerate(word_probs) if prob > 0)
                top.sort(key=lambda item: (-item[0], item[1]))
                for prob, token in top[:beam_width]:
                    child = {
                        "h": parent["h"].copy(), "prev": parent["prev"].copy(),
                        "cov": parent["cov"].copy(), "score": parent["score"] + np.log(prob + 1e-30),
                        "tokens": parent["tokens"] + 1, "done": token == 1,
                        "title": parent["title"],
                    }
                    if token != 1:
                        if token < V:
                            append_generated(child, token)
                            child["prev"] = self.tensors["emb.weight"][token]
                        else:
                            group = token - V
                            append_copied(child, group)
                            positions = [i for i, g in enumerate(token_word) if g == group]
                            child["prev"] = self.tensors["emb.weight"][[sel_ids[i] for i in positions]].mean(axis=0)
                        dec_input = child["prev"] + np.tanh(self.tensors["dec_ctx.weight"] @ ctx)
                        child["h"] = self._gru("dec", dec_input, parent["h"])
                        child["cov"] = parent["cov"] + attn
                    candidates.append(child)
            if not candidates:
                break
            candidates.sort(key=rank, reverse=True)
            beams = candidates[:beam_width]
            if all(hyp["done"] for hyp in beams):
                break
        winner = max(beams, key=rank)
        words, seen = [], set()
        small = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
                 "into", "nor", "of", "on", "or", "over", "per", "the", "to", "via",
                 "vs", "with"}
        for word in winner["title"].split():
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            internal_upper = any(c.isupper() for c in word[1:])
            if key in small and words:
                word = word.lower()
            elif not internal_upper:
                lowered = word.lower()
                for i, char in enumerate(lowered):
                    if char.isalpha():
                        word = lowered[:i] + char.upper() + lowered[i + 1:]
                        break
            words.append(word)
        return " ".join(words)


def main():
    path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "hello world"
    m = TTM1(path)
    print(m.decode(text))


if __name__ == "__main__":
    main()
