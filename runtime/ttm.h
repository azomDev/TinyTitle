// ttm.h — .ttm1 loader + tokenizer + forward for the TinyTitle pointer-generator
// Single-header, bounded, no dynamic allocation after init.
#ifndef TTM_H
#define TTM_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define TTM_MAGIC "TTM1"
#define TTM_TAG_VOCAB 1
#define TTM_TAG_WEIGHTS 2

#define TTM_MAX_VOCAB 8000
#define TTM_MAX_SRC 256
#define TTM_MAX_TGT 16
#define TTM_MAX_BYTES 6000
#define TTM_MAX_PIECES 2048
#define TTM_MAX_TENSORS 64
#define TTM_MAX_NAME 64
#define TTM_D_EMB 128
#define TTM_D_ENC 128
#define TTM_D_DEC 256
#define TTM_D_ATT 160
#define TTM_V 8000

// vocabulary entry layout (export.py): u8 blen, bytes, u16 zero, f32 score
typedef struct {
    uint32_t offset;
    uint32_t size;
    uint8_t tag;
} ttm_section;

typedef struct {
    // file mapping
    const uint8_t *data;
    size_t file_size;
    ttm_section sections[TTM_MAX_TENSORS];
    int n_sections;
    // vocab
    const uint8_t *vocab;   // points at count field
    uint32_t vocab_count;
    // tensors (name -> offset)
    struct {
        char name[TTM_MAX_NAME];
        int dims;
        int32_t d0, d1;
        uint64_t row_bytes;  // 0 = raw f32 rows (2-D) or 1-D
        const uint8_t *data;
    } tensors[TTM_MAX_TENSORS];
    int n_tensors;
} ttm_model;

// ---- safe unaligned reads (the file is packed; fields may not be aligned) ----
static inline uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t rd_u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline int32_t rd_i32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }

// ---- loader ----
static int ttm_load(const char *path, ttm_model *m) {
    memset(m, 0, sizeof(*m));
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return -1; }
    m->file_size = (size_t)st.st_size;
    m->data = (const uint8_t *)mmap(NULL, m->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->data == MAP_FAILED) { perror("mmap"); return -1; }

    const uint8_t *p = m->data;
    if (m->file_size < 4 + 4 + 4 + 8) { fprintf(stderr, "ttm1: file too small\n"); return -1; }
    if (memcmp(p, TTM_MAGIC, 4) != 0) { fprintf(stderr, "ttm1: bad magic\n"); return -1; }
    uint32_t n_vocab = rd_u32(p + 4);
    uint32_t n_sections = rd_u32(p + 8);
    uint64_t table_off = rd_u64(p + 12);
    if (n_vocab > TTM_MAX_VOCAB || n_sections > TTM_MAX_TENSORS) {
        fprintf(stderr, "ttm1: limits exceeded\n");
        return -1;
    }
    if (table_off + (uint64_t)n_sections * 24 > m->file_size) {
        fprintf(stderr, "ttm1: bad section table\n");
        return -1;
    }
    const uint8_t *tbl = m->data + table_off;
    m->n_sections = (int)n_sections;
    for (uint32_t i = 0; i < n_sections; i++) {
        uint32_t tag = rd_u32(tbl + i * 24);
        uint64_t off = rd_u64(tbl + i * 24 + 8);
        uint64_t sz = rd_u64(tbl + i * 24 + 16);
        if (off + sz > m->file_size) { fprintf(stderr, "ttm1: section out of range\n"); return -1; }
        m->sections[i].tag = (uint8_t)tag;
        m->sections[i].offset = (uint32_t)off;
        m->sections[i].size = (uint32_t)sz;
    }
    // vocab section
    for (int i = 0; i < m->n_sections; i++) {
        if (m->sections[i].tag == TTM_TAG_VOCAB) {
            m->vocab = m->data + m->sections[i].offset;
            m->vocab_count = rd_u32(m->vocab);
            if (m->vocab_count > TTM_MAX_VOCAB) { fprintf(stderr, "ttm1: vocab too big\n"); return -1; }
        }
    }
    // weights section: parse the tensor list
    for (int i = 0; i < m->n_sections; i++) {
        if (m->sections[i].tag != TTM_TAG_WEIGHTS) continue;
        const uint8_t *w = m->data + m->sections[i].offset;
        const uint8_t *wend = w + m->sections[i].size;
        uint32_t count = rd_u32(w);
        if (count > TTM_MAX_TENSORS) { fprintf(stderr, "ttm1: too many tensors\n"); return -1; }
        w += 4;
        for (uint32_t k = 0; k < count && w + 4 <= wend; k++) {
            uint32_t nl = rd_u32(w); w += 4;
            if (nl >= TTM_MAX_NAME || w + nl > wend) return -1;
            if (m->n_tensors >= TTM_MAX_TENSORS) return -1;
            ttm_model *mm = m;
            memcpy(mm->tensors[m->n_tensors].name, w, nl);
            mm->tensors[m->n_tensors].name[nl] = 0;
            w += nl;
            // names are padded to 4-byte alignment
            w += (4 - ((w - m->data) & 3)) & 3;
            uint32_t dims = rd_u32(w); w += 4;
            int32_t a = rd_i32(w); w += 4;
            int32_t b = rd_i32(w); w += 4;
            mm->tensors[m->n_tensors].dims = (int)dims;
            mm->tensors[m->n_tensors].d0 = a;
            mm->tensors[m->n_tensors].d1 = b;
            mm->tensors[m->n_tensors].row_bytes = 0;
            m->n_tensors++;
            if (getenv("TTM_DEBUG")) {
                fprintf(stderr, "[load] %s dims=%u %dx%d file_off=%ld\n",
                        mm->tensors[m->n_tensors - 1].name, dims, a, b,
                        (long)(w - m->data));
            }
            if (dims == 2) {
                uint64_t rows = rd_u64(w); w += 8;
                uint64_t row_bytes = rd_u64(w); w += 8;
                if (rows > 1u << 20 || row_bytes > 1u << 20) return -1;
                mm->tensors[m->n_tensors - 1].row_bytes = row_bytes;
                if (row_bytes == 0) {
                    // raw f32 rows: a*b floats
                    if ((size_t)a * b > (size_t)(wend - w) / 4) return -1;
                    mm->tensors[m->n_tensors - 1].data = w;
                    w += (size_t)a * b * 4;
                } else {
                    if (w + 4 * rows + row_bytes > wend) return -1;
                    mm->tensors[m->n_tensors - 1].data = w;
                    w += 4 * rows + row_bytes;
                }
            } else {
                if (a < 0 || (size_t)a * 4 > (size_t)(wend - w)) return -1;
                mm->tensors[m->n_tensors - 1].data = w;
                w += (size_t)a * 4;
            }
        }
    }
    return 0;
}

static void ttm_unload(ttm_model *m) {
    if (m->data && m->data != MAP_FAILED) munmap((void *)m->data, m->file_size);
    memset(m, 0, sizeof(*m));
}

// touch every page of the mapping (the plan requires all-pages-touched RSS)
static void ttm_touch_all(const ttm_model *m) {
    volatile uint8_t acc = 0;
    for (size_t off = 0; off < m->file_size; off += 4096) {
        acc ^= m->data[off];
    }
    (void)acc;
}

// find tensor by name
static int ttm_find_tensor(const ttm_model *m, const char *name) {
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) return i;
    return -1;
}

static const uint8_t *ttm_tensor(const ttm_model *m, const char *name, int *dims,
                                 int32_t *d0, int32_t *d1) {
    int i = ttm_find_tensor(m, name);
    if (i >= 0) {
        if (dims) *dims = m->tensors[i].dims;
        if (d0) *d0 = m->tensors[i].d0;
        if (d1) *d1 = m->tensors[i].d1;
        return m->tensors[i].data;
    }
    fprintf(stderr, "ttm1: tensor %s not found\n", name);
    return NULL;
}

// dequantize row r of a 2-D tensor into out[cols] (f32)
// int8 layout: [f32 scale x d0] [int8 rows x row_bytes]; f32 layout: [f32 d0*d1]
static void ttm_row(const ttm_model *m, const char *name, int r, float *out) {
    int dims; int32_t d0, d1;
    const uint8_t *t = ttm_tensor(m, name, &dims, &d0, &d1);
    if (!t || dims != 2) { fprintf(stderr, "ttm1: bad tensor %s\n", name); return; }
    if (r < 0 || r >= d0) { fprintf(stderr, "ttm1: row %d out of range for %s (%dx%d)\n", r, name, d0, d1); return; }
    uint64_t row_bytes = 0;
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) { row_bytes = m->tensors[i].row_bytes; break; }
    if (row_bytes == 0) {
        memcpy(out, t + (size_t)r * d1 * 4, (size_t)d1 * 4);
        return;
    }
    const float *scales = (const float *)t;
    const int8_t *q = (const int8_t *)(t + 4 * (size_t)d0);
    float s = scales[r];
    size_t row_stride = row_bytes / (size_t)d0;  // stored row_bytes = d0*d1 total
    const int8_t *qr = q + (size_t)r * row_stride;
    for (int i = 0; i < d1; i++) out[i] = s * (float)qr[i];
}

// direct matrix-vector multiply without materializing dequantized rows.
static int ttm_matvec(const ttm_model *m, const char *name, int rows, int cols,
                      const float *x, float *y) {
    int ti = ttm_find_tensor(m, name);
    if (ti < 0) return -1;
    const int d0 = m->tensors[ti].d0, d1 = m->tensors[ti].d1;
    const uint8_t *t = m->tensors[ti].data;
    if (m->tensors[ti].dims != 2 || d0 != rows || d1 != cols) return -1;
    uint64_t row_bytes = m->tensors[ti].row_bytes;
    if (row_bytes == 0) {
        const float *w = (const float *)t;
        for (int r = 0; r < rows; r++) {
            float acc = 0.f;
            for (int c = 0; c < cols; c++) acc += w[(size_t)r * cols + c] * x[c];
            y[r] = acc;
        }
        return 0;
    }
    const float *scales = (const float *)t;
    const int8_t *q = (const int8_t *)(t + 4 * (size_t)d0);
    size_t stride = row_bytes / (size_t)d0;
    for (int r = 0; r < rows; r++) {
        const int8_t *qr = q + (size_t)r * stride;
        float acc = 0.f;
        for (int c = 0; c < cols; c++) acc += (float)qr[c] * x[c];
        y[r] = acc * scales[r];
    }
    return 0;
}

// get a scalar of a 2-D tensor
static float ttm_elem2(const ttm_model *m, const char *name, int r, int c) {
    int dims; int32_t d0, d1;
    const uint8_t *t = ttm_tensor(m, name, &dims, &d0, &d1);
    if (!t || dims != 2 || r >= d0 || c >= d1) return 0.f;
    uint64_t row_bytes = 0;
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) { row_bytes = m->tensors[i].row_bytes; break; }
    if (row_bytes == 0) {
        return ((const float *)t)[(size_t)r * d1 + c];
    }
    const float *scales = (const float *)t;
    const int8_t *q = (const int8_t *)(t + 4 * (size_t)d0);
    size_t row_stride = row_bytes / (size_t)d0;
    return scales[r] * (float)q[(size_t)r * row_stride + c];
}

// 1-D f32 tensor
static const float *ttm_vec(const ttm_model *m, const char *name, int *len) {
    int dims; int32_t d0, d1;
    const uint8_t *t = ttm_tensor(m, name, &dims, &d0, &d1);
    if (!t || dims != 1) { fprintf(stderr, "ttm1: bad vec %s\n", name); return NULL; }
    if (len) *len = d0;
    return (const float *)t;
}

// ---- vocab ----
// entry: u8 blen, bytes, u16 zero, f32 score
static const uint8_t *ttm_vocab_entry(const ttm_model *m, uint32_t id) {
    if (id >= m->vocab_count) return NULL;
    const uint8_t *p = m->vocab + 4;
    for (uint32_t i = 0; i < id; i++) {
        uint8_t bl = *p;
        p += 1 + bl + 2 + 4;
    }
    return p;
}

static const char *ttm_vocab_str(const ttm_model *m, uint32_t id, int *len, float *score) {
    const uint8_t *p = ttm_vocab_entry(m, id);
    if (!p) { *len = 0; if (score) *score = -20.f; return ""; }
    uint8_t bl = *p;
    if (score) { memcpy(score, p + 1 + bl + 2, 4); }
    *len = bl;
    return (const char *)(p + 1);
}


// ---- tokenizer: unigram viterbi over metaspace-prefixed whitespace pieces ----
#define TTM_UNK_ID 2

static int ttm_is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static int ttm_utf8_len(const uint8_t *p, int remaining) {
    if (remaining <= 0) return 0;
    if (p[0] < 0x80) return 1;
    int n = (p[0] >= 0xc2 && p[0] <= 0xdf) ? 2 :
            (p[0] >= 0xe0 && p[0] <= 0xef) ? 3 :
            (p[0] >= 0xf0 && p[0] <= 0xf4) ? 4 : 0;
    if (!n || n > remaining) return 0;
    for (int i = 1; i < n; i++) if ((p[i] & 0xc0) != 0x80) return 0;
    if (n == 3 && p[0] == 0xe0 && p[1] < 0xa0) return 0;
    if (n == 3 && p[0] == 0xed && p[1] >= 0xa0) return 0;
    if (n == 4 && p[0] == 0xf0 && p[1] < 0x90) return 0;
    if (n == 4 && p[0] == 0xf4 && p[1] >= 0x90) return 0;
    return n;
}

// viterbi over one byte buffer; returns token count, fills ids + byte spans
static int ttm_viterbi(const ttm_model *m, const uint8_t *buf, int blen,
                       uint32_t *ids, int *byte_start, int *byte_end, int max_tokens) {
    if (blen <= 0) return 0;
    static int32_t dp[6004];
    static int32_t prev[6004];
    static uint16_t tok[6004];
    const int NEG = INT32_MIN / 4;
    const int Q = 1 << 12;
    dp[0] = 0; prev[0] = -1;
    for (int i = 1; i <= blen; i++) dp[i] = NEG;
    for (int i = 0; i < blen; i++) {
        if (dp[i] == NEG) continue;
        const uint8_t *p = m->vocab + 4;
        for (uint32_t v = 0; v < m->vocab_count; v++) {
            uint8_t l = *p;
            const uint8_t *s = p + 1;
            float score;
            memcpy(&score, p + 1 + l + 2, 4);
            if (i + l <= blen && memcmp(s, buf + i, l) == 0) {
                int32_t sc = dp[i] + (int32_t)(score * Q);
                if (sc > dp[i + l]) {
                    dp[i + l] = sc;
                    prev[i + l] = i;
                    tok[i + l] = (uint16_t)v;
                }
            }
            p += 1 + l + 2 + 4;
        }
        // unknown fallback consumes one complete utf-8 code point, matching
        // tokenizers and ensuring copied unknowns remain valid utf-8.
        {
            int cp = ttm_utf8_len(buf + i, blen - i);
            if (cp <= 0) cp = 1;
            int32_t sc = dp[i] + (int32_t)(-20.0f * Q);
            if (sc > dp[i + cp]) {
                dp[i + cp] = sc;
                prev[i + cp] = i;
                tok[i + cp] = TTM_UNK_ID;
            }
        }
    }
    if (dp[blen] == NEG) return 0;
    int pos = blen, n = 0;
    while (pos > 0 && n < max_tokens) {
        int p0 = prev[pos];
        ids[n] = tok[pos];
        byte_start[n] = p0;
        byte_end[n] = pos;
        n++;
        pos = p0;
    }
    for (int i = 0; i < n / 2; i++) {
        uint32_t t = ids[i]; ids[i] = ids[n - 1 - i]; ids[n - 1 - i] = t;
        int s = byte_start[i]; byte_start[i] = byte_start[n - 1 - i]; byte_start[n - 1 - i] = s;
        int e = byte_end[i]; byte_end[i] = byte_end[n - 1 - i]; byte_end[n - 1 - i] = e;
    }
    return n;
}

// tokenizers' metaspace pre-tokenizer replaces each whitespace boundary with ▁
// and prepends one at the start. process each word independently to avoid a
// second input-sized workspace while preserving identical unigram pieces.
static int ttm_tokenize(const ttm_model *m, const uint8_t *buf, int blen,
                        uint32_t *ids, int *byte_start, int *byte_end, int max_tokens) {
    int n = 0, i = 0;
    while (i < blen && n < max_tokens) {
        while (i < blen && ttm_is_ws(buf[i])) i++;
        if (i >= blen) break;
        int j = i;
        while (j < blen && !ttm_is_ws(buf[j])) j++;
        int word_len = j - i;
        uint8_t piece[TTM_MAX_BYTES + 3];
        piece[0] = 0xe2; piece[1] = 0x96; piece[2] = 0x81; // utf-8 ▁
        memcpy(piece + 3, buf + i, (size_t)word_len);
        uint32_t pids[TTM_MAX_SRC * 4];
        int pst[TTM_MAX_SRC * 4], pen[TTM_MAX_SRC * 4];
        int cap = max_tokens - n;
        if (cap > TTM_MAX_SRC * 4) cap = TTM_MAX_SRC * 4;
        int k = ttm_viterbi(m, piece, word_len + 3, pids, pst, pen, cap);
        for (int t = 0; t < k && n < max_tokens; t++) {
            ids[n] = pids[t];
            int s = pst[t] - 3, e = pen[t] - 3;
            if (s < 0) s = 0;
            if (e < 0) e = 0;
            if (s > word_len) s = word_len;
            if (e > word_len) e = word_len;
            byte_start[n] = i + s;
            byte_end[n] = i + e;
            n++;
        }
        i = j;
    }
    return n;
}

#endif
