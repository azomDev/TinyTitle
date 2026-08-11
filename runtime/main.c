// main.c — TinyTitle pointer-generator title CLI (.ttm1)
// Usage: ./title-v1 <model.ttm1> "user message"   (or stdin)
#include "ttm.h"
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>

// ---- tensor name constants (match python state_dict) ----
#define T_EMB      "emb.weight"
#define T_ENC_F_W  "enc.fwd.w.weight"
#define T_ENC_F_U  "enc.fwd.uh.weight"
#define T_ENC_F_B  "enc.fwd.b"
#define T_ENC_R_W  "enc.bwd.w.weight"
#define T_ENC_R_U  "enc.bwd.uh.weight"
#define T_ENC_R_B  "enc.bwd.b"
#define T_DEC_W    "dec.w.weight"
#define T_DEC_CTX  "dec_ctx.weight"
#define T_DEC_U    "dec.uh.weight"
#define T_DEC_B    "dec.b"
#define T_WATT_V   "w_att_v.weight"
#define T_WATT_H   "w_att_h.weight"
#define T_V_ATT    "v_att.weight"
#define T_COV      "cov.weight"
#define T_WC       "w_c.weight"
#define T_WO       "w_o.weight"
#define T_GATE     "gate.weight"
#define T_GATE_B   "gate.bias"
#define T_DEC_INIT "dec_init.weight"
#define T_DEC_INIT_B "dec_init.bias"

#define D_EMB TTM_D_EMB
#define D_ENC_HALF TTM_D_ENC
#define D_ENC (2 * TTM_D_ENC)
#define D_DEC TTM_D_DEC
#define D_ATT TTM_D_ATT
#define V_MAX TTM_MAX_VOCAB


// ---- matvec: [rows x cols] int8 tensor * x -> y ----
static void matvec(const ttm_model *m, const char *name, int rows, int cols,
                   const float *x, float *y, float *rowbuf) {
    (void)rowbuf;
    if (ttm_matvec(m, name, rows, cols, x, y) != 0) {
        fprintf(stderr, "ttm1: invalid matvec tensor %s\n", name);
        memset(y, 0, (size_t)rows * sizeof(float));
    }
}

// ---- GRU cell (matches python GRUCell) ----
// x: [in], h: [hid] -> h_out [hid]; scratch needs 6*hid floats
static void gru_cell(const ttm_model *m, const char *w_name, const char *u_name,
                     const char *b_name, int in, int hid, const float *x, const float *h,
                     float *h_out, float *scratch) {
    float *gx = scratch;                // 3*hid
    float *gh = scratch + 3 * hid;      // 3*hid
    float *rowbuf = scratch + 6 * hid;  // max(in, hid) floats
    matvec(m, w_name, 3 * hid, in, x, gx, rowbuf);
    matvec(m, u_name, 3 * hid, hid, h, gh, rowbuf);
    const float *bias = ttm_vec(m, b_name, NULL);
    for (int i = 0; i < 3 * hid; i++) gx[i] += bias[i];
    for (int i = 0; i < hid; i++) {
        float rr = 1.f / (1.f + expf(-(gx[i] + gh[i])));
        float zz = 1.f / (1.f + expf(-(gx[hid + i] + gh[hid + i])));
        float nn = tanhf(gx[2 * hid + i] + rr * gh[2 * hid + i]);
        h_out[i] = (1.f - zz) * nn + zz * h[i];
    }
}

// ---- encode: fills src_enc [n][2*d_enc] ----
static void encode(const ttm_model *m, const uint32_t *ids, int n,
                   float *src_enc, float *work) {
    float *emb = work;
    float *hf = work + (size_t)n * D_EMB;
    float *hb = hf + D_ENC_HALF;
    float *tmp = hb + D_ENC_HALF;
    float *gscratch = tmp + D_ENC_HALF;
    for (int i = 0; i < n; i++)
        ttm_row(m, T_EMB, (int)ids[i], emb + (size_t)i * D_EMB);
    memset(hf, 0, D_ENC_HALF * sizeof(float));
    memset(hb, 0, D_ENC_HALF * sizeof(float));
    for (int i = 0; i < n; i++) {
        gru_cell(m, T_ENC_F_W, T_ENC_F_U, T_ENC_F_B,
                 D_EMB, D_ENC_HALF, emb + (size_t)i * D_EMB, hf, tmp, gscratch);
        memcpy(hf, tmp, D_ENC_HALF * sizeof(float));
        memcpy(src_enc + (size_t)i * D_ENC, hf, D_ENC_HALF * sizeof(float));
    }
    for (int i = n - 1; i >= 0; i--) {
        gru_cell(m, T_ENC_R_W, T_ENC_R_U, T_ENC_R_B,
                 D_EMB, D_ENC_HALF, emb + (size_t)i * D_EMB, hb, tmp, gscratch);
        memcpy(hb, tmp, D_ENC_HALF * sizeof(float));
        memcpy(src_enc + (size_t)i * D_ENC + D_ENC_HALF,
               hb, D_ENC_HALF * sizeof(float));
    }
}

// ---- decode one step ----
// h: [192] decoder state, prev_emb: [96], cov: [n] coverage vector
// fills attn [n], logits [V], p_gen [1]
// scratch needs: n*128 (e) + 192 (ctx) + 96 (o) + 96 (logit_in) + max(rows) rowbuf
static void decode_step(const ttm_model *m, const float *h, const float *prev_emb,
                        const float *src_enc, int n,
                        const float *cov, float *attn, float *logits, float *p_gen,
                        float *ctx_out, float *scratch) {
    float *e = scratch;                       // n*128
    float *ctx = scratch + (size_t)n * D_ATT; // 192
    float *o = ctx + D_ENC;                   // 96
    float *li = o + D_EMB;                    // 96
    float *rowbuf = li + D_EMB;               // max(192,128,96,3*192)

    float *wh = rowbuf;
    matvec(m, T_WATT_H, D_ATT, D_DEC, h, wh, rowbuf + D_ATT);
    float *vr = rowbuf + 2 * D_ATT;
    // e[i] = w_att_v(src_enc[i]) + wh + cov[i]*w_cov
    // w_att_v: [128 x 192]
    for (int i = 0; i < n; i++) {
        matvec(m, T_WATT_V, D_ATT, D_ENC, src_enc + (size_t)i * D_ENC, e + (size_t)i * D_ATT, vr);
        for (int j = 0; j < D_ATT; j++) e[(size_t)i * D_ATT + j] += wh[j];
        // cov term: cov is [1] per position; skip (python adds cov via cov.weight)
        // cov.weight: [128 x 1] — c[j] = cov.weight[j] * cov[i]
        for (int j = 0; j < D_ATT; j++) {
            float cw = ttm_elem2(m, T_COV, j, 0);
            e[(size_t)i * D_ATT + j] += cw * cov[i];
        }
    }
    // scores = v_att(tanh(e)) -> [n]
    float mx = -1e30f;
    for (int i = 0; i < n; i++) {
        float s = 0.f;
        float *ei = e + (size_t)i * D_ATT;
        for (int j = 0; j < D_ATT; j++) {
            float vw = ttm_elem2(m, T_V_ATT, 0, j);
            s += vw * tanhf(ei[j]);
        }
        attn[i] = s;
        if (s > mx) mx = s;
    }
    // softmax over n
    float sum = 0.f;
    for (int i = 0; i < n; i++) { attn[i] = expf(attn[i] - mx); sum += attn[i]; }
    for (int i = 0; i < n; i++) attn[i] /= sum;
    // ctx = sum attn[i] * src_enc[i]
    memset(ctx, 0, D_ENC * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < D_ENC; j++)
            ctx[j] += attn[i] * src_enc[(size_t)i * D_ENC + j];
    memcpy(ctx_out, ctx, D_ENC * sizeof(float));
    const int context_in = D_DEC + D_ENC;
    const int output_in = D_DEC + D_EMB;
    const int gate_in = D_DEC + D_ENC + D_EMB;
    float *hc = rowbuf;
    memcpy(hc, h, D_DEC * sizeof(float));
    memcpy(hc + D_DEC, ctx, D_ENC * sizeof(float));
    float *rowbuf2 = hc + gate_in;
    matvec(m, T_WC, D_EMB, context_in, hc, li, rowbuf2);
    memcpy(hc, h, D_DEC * sizeof(float));
    memcpy(hc + D_DEC, prev_emb, D_EMB * sizeof(float));
    matvec(m, T_WO, D_EMB, output_in, hc, o, rowbuf2);
    for (int i = 0; i < D_EMB; i++) li[i] += o[i];
    // logits = head(tanh(li)) ; head tied to emb: logits[v] = emb[v] . tanh(li)
    float *tanh_li = rowbuf2;                 // 96
    for (int i = 0; i < D_EMB; i++) tanh_li[i] = tanhf(li[i]);
    if (getenv("TTM_DEBUG")) {
        fprintf(stderr, "[dbg] li[:4]=%.6f %.6f %.6f %.6f ctx[:3]=%.6f %.6f %.6f\n",
                li[0], li[1], li[2], li[3], ctx[0], ctx[1], ctx[2]);
    }
    for (int v = 0; v < V_MAX; v++) logits[v] = 0.f;
    matvec(m, T_EMB, (int)m->vocab_count, D_EMB, tanh_li, logits, NULL);
    memcpy(hc, h, D_DEC * sizeof(float));
    memcpy(hc + D_DEC, ctx, D_ENC * sizeof(float));
    memcpy(hc + D_DEC + D_ENC, prev_emb, D_EMB * sizeof(float));
    const float *gate_bias = ttm_vec(m, T_GATE_B, NULL);
    float g = gate_bias[0];
    for (int j = 0; j < gate_in; j++) g += ttm_elem2(m, T_GATE, 0, j) * hc[j];
    *p_gen = 1.f / (1.f + expf(-g));

    if (getenv("TTM_DEBUG")) {
        // top 5
        int top5[5]; float topv[5];
        for (int k = 0; k < 5; k++) { top5[k] = -1; topv[k] = -1e30f; }
        for (int v = 0; v < (int)m->vocab_count; v++) {
            for (int k = 0; k < 5; k++) {
                if (logits[v] > topv[k]) {
                    for (int kk = 4; kk > k; kk--) { top5[kk] = top5[kk-1]; topv[kk] = topv[kk-1]; }
                    top5[k] = v; topv[k] = logits[v];
                    break;
                }
            }
        }
        fprintf(stderr, "[dbg] p_gen=%f attn0=%f top:", *p_gen, attn[0]);
        for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.5f)", top5[k], topv[k]);
        fprintf(stderr, "\n");
    }
    if (getenv("TTM_LOGITS")) {
        fprintf(stderr, "[logits]");
        for (int v = 0; v < 64 && v < (int)m->vocab_count; v++) fprintf(stderr, " %.6f", logits[v]);
        fprintf(stderr, "\n");
    }
}

static void vocab_probs(float *probs, const float *logits, float p_gen, float temp, int V) {
    float mx = -1e30f;
    for (int v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    float sum = 0.f;
    for (int v = 0; v < V; v++) {
        probs[v] = expf((logits[v] - mx) / temp);
        sum += probs[v];
    }
    for (int v = 0; v < V; v++) probs[v] = p_gen * probs[v] / sum;
}

// render a token to output text; returns bytes written
// copied tokens render from the source span (original case), generated from vocab
static uint32_t utf8_codepoint(const uint8_t *p, int n) {
    if (n == 1) return p[0];
    if (n == 2) return ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    if (n == 3) return ((uint32_t)(p[0] & 0x0f) << 12) |
                       ((uint32_t)(p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    return ((uint32_t)(p[0] & 0x07) << 18) |
           ((uint32_t)(p[1] & 0x3f) << 12) |
           ((uint32_t)(p[2] & 0x3f) << 6) | (p[3] & 0x3f);
}

static int lexical_codepoint(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
           (cp >= '0' && cp <= '9') || cp == '_' || (cp >= 0x00c0 && cp <= 0x024f);
}

static int lexical_connector(uint32_t cp) {
    return cp == '\'' || cp == '-' || cp == '.' || cp == 0x2019;
}

static int lexical_spans(const char *text, int *starts, int *ends, int max_words) {
    const uint8_t *s = (const uint8_t *)text;
    int len = (int)strlen(text), n = 0, i = 0;
    while (i < len && n < max_words) {
        int cp_len = ttm_utf8_len(s + i, len - i);
        uint32_t cp = utf8_codepoint(s + i, cp_len);
        int leading_dot = cp == '.';
        if (leading_dot) {
            int next = i + cp_len;
            if (next >= len) { i += cp_len; continue; }
            int next_len = ttm_utf8_len(s + next, len - next);
            if (!lexical_codepoint(utf8_codepoint(s + next, next_len))) {
                i += cp_len;
                continue;
            }
        } else if (!lexical_codepoint(cp)) {
            i += cp_len;
            continue;
        }
        int start = i;
        i += cp_len;
        while (i < len) {
            cp_len = ttm_utf8_len(s + i, len - i);
            cp = utf8_codepoint(s + i, cp_len);
            if (lexical_codepoint(cp)) { i += cp_len; continue; }
            if (cp == '+' || cp == '#') { i += cp_len; continue; }
            if (lexical_connector(cp)) {
                int next = i + cp_len;
                if (next < len) {
                    int next_len = ttm_utf8_len(s + next, len - next);
                    uint32_t next_cp = utf8_codepoint(s + next, next_len);
                    if (lexical_codepoint(next_cp)) { i = next; continue; }
                }
            }
            break;
        }
        starts[n] = start;
        ends[n] = i;
        n++;
    }
    return n;
}

static int valid_input_bytes(const char *text) {
    const uint8_t *p = (const uint8_t *)text;
    int remaining = (int)strlen(text);
    while (remaining > 0) {
        if (*p < 0x20 && !ttm_is_ws(*p)) return 0;
        int n = ttm_utf8_len(p, remaining);
        if (n <= 0) return 0;
        p += n;
        remaining -= n;
    }
    return 1;
}

static int ascii_word_eq(const char *a, const char *b, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t x = (uint8_t)a[i], y = (uint8_t)b[i];
        if (x >= 'A' && x <= 'Z') x += 'a' - 'A';
        if (y >= 'A' && y <= 'Z') y += 'a' - 'A';
        if (x != y) return 0;
    }
    return 1;
}

static void collapse_duplicate_words(char *s) {
    char out[512];
    int oi = 0;
    for (int i = 0; s[i] && oi < (int)sizeof(out) - 1;) {
        while (s[i] == ' ') i++;
        if (!s[i]) break;
        int start = i;
        while (s[i] && s[i] != ' ') i++;
        int len = i - start, duplicate = 0;
        for (int p = 0; p < oi;) {
            while (p < oi && out[p] == ' ') p++;
            int q = p;
            while (q < oi && out[q] != ' ') q++;
            if (q - p == len && ascii_word_eq(out + p, s + start, len)) {
                duplicate = 1;
                break;
            }
            p = q;
        }
        if (duplicate) continue;
        if (oi > 0) out[oi++] = ' ';
        memcpy(out + oi, s + start, (size_t)len);
        oi += len;
    }
    out[oi] = 0;
    memcpy(s, out, (size_t)oi + 1);
}

static int ascii_small_word(const char *s, int len) {
    static const char *small[] = {
        "a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
        "into", "nor", "of", "on", "or", "over", "per", "the", "to", "via",
        "vs", "with"
    };
    for (size_t k = 0; k < sizeof(small) / sizeof(small[0]); k++) {
        int n = (int)strlen(small[k]);
        if (n != len) continue;
        int same = 1;
        for (int i = 0; i < len; i++) {
            uint8_t c = (uint8_t)s[i];
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
            if (c != (uint8_t)small[k][i]) { same = 0; break; }
        }
        if (same) return 1;
    }
    return 0;
}

static void title_case_ascii(char *s) {
    int word_index = 0;
    for (int i = 0; s[i];) {
        while (s[i] == ' ') i++;
        if (!s[i]) break;
        int start = i;
        while (s[i] && s[i] != ' ') i++;
        int end = i, internal_upper = 0, first_alpha = -1;
        for (int j = start; j < end; j++) {
            uint8_t c = (uint8_t)s[j];
            if (first_alpha < 0 && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
                first_alpha = j;
            if (j > start && c >= 'A' && c <= 'Z') internal_upper = 1;
        }
        int small = word_index > 0 && ascii_small_word(s + start, end - start);
        if (small || !internal_upper) {
            for (int j = start; j < end; j++)
                if (s[j] >= 'A' && s[j] <= 'Z') s[j] += 'a' - 'A';
        }
        if (!small && !internal_upper && first_alpha >= 0 &&
            s[first_alpha] >= 'a' && s[first_alpha] <= 'z')
            s[first_alpha] -= 'a' - 'A';
        word_index++;
    }
}

static int render_token(const ttm_model *m, uint32_t id, char *out) {
    int len;
    const char *s = ttm_vocab_str(m, id, &len, NULL);
    int written = 0;
    for (int i = 0; i < len;) {
        if (i + 2 < len && (uint8_t)s[i] == 0xe2 &&
            (uint8_t)s[i + 1] == 0x96 && (uint8_t)s[i + 2] == 0x81) {
            out[written++] = ' ';
            i += 3;
        } else {
            out[written++] = s[i++];
        }
    }
    return written;
}

#define TTM_BEAM 2

typedef struct {
    float h[D_DEC];
    float prev_emb[D_EMB];
    float cov[TTM_MAX_SRC];
    float score;
    int tokens;
    int done;
    char title[512];
    int title_len;
} beam_hyp;

static int rendered_word_count(const char *s) {
    int count = 0, in_word = 0;
    for (; *s; s++) {
        int ws = *s == ' ' || *s == '\t' || *s == '\n' || *s == '\r';
        if (!ws && !in_word) { count++; in_word = 1; }
        if (ws) in_word = 0;
    }
    return count;
}

static float beam_rank(const beam_hyp *b) {
    if (b->done && rendered_word_count(b->title) < 2) return -1e30f;
    float length_penalty = powf((5.f + (float)(b->tokens > 0 ? b->tokens : 1)) / 6.f, 0.6f);
    return b->score / length_penalty;
}

static void append_generated(beam_hyp *b, const ttm_model *m, uint32_t id) {
    char rendered[256];
    int len = render_token(m, id, rendered);
    int skip = b->title_len == 0 && len > 0 && rendered[0] == ' ';
    if (skip) { len--; memmove(rendered, rendered + 1, (size_t)len); }
    if (len > (int)sizeof(b->title) - 1 - b->title_len)
        len = (int)sizeof(b->title) - 1 - b->title_len;
    memcpy(b->title + b->title_len, rendered, (size_t)len);
    b->title_len += len;
    b->title[b->title_len] = 0;
}

static void append_copied_word(beam_hyp *b, const char *text, int bs, int be) {
    if (b->title_len > 0 && b->title[b->title_len - 1] != ' ' &&
        b->title_len < (int)sizeof(b->title) - 1) b->title[b->title_len++] = ' ';
    int len = be - bs;
    if (len > (int)sizeof(b->title) - 1 - b->title_len)
        len = (int)sizeof(b->title) - 1 - b->title_len;
    memcpy(b->title + b->title_len, text + bs, (size_t)len);
    b->title_len += len;
    b->title[b->title_len] = 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.ttm1> [\"user message\"]\n", argv[0]); return 1; }
    ttm_model m;
    if (ttm_load(argv[1], &m) != 0) return 1;
    int emb_dims = 0; int32_t emb_rows = 0, emb_cols = 0;
    ttm_tensor(&m, T_EMB, &emb_dims, &emb_rows, &emb_cols);
    if (ttm_find_tensor(&m, T_DEC_CTX) < 0 || emb_dims != 2 ||
        emb_rows != (int32_t)m.vocab_count || emb_cols != D_EMB) {
        fprintf(stderr, "incompatible ttm1: expected word-copy dimensions\n");
        ttm_unload(&m);
        return 1;
    }
    if (getenv("TTM_TOUCH")) ttm_touch_all(&m);

    const char *text = (argc >= 3) ? argv[2] : NULL;
    char buf[TTM_MAX_BYTES + 2];
    if (!text) {
        if (!fgets(buf, sizeof(buf), stdin)) return 1;
        buf[strcspn(buf, "\n")] = 0;
        text = buf;
    }
    if (strlen(text) > TTM_MAX_BYTES) { fprintf(stderr, "input too long\n"); return 1; }
    if (!valid_input_bytes(text)) { fprintf(stderr, "input contains control bytes\n"); return 1; }

    // tokenize
    uint32_t ids[TTM_MAX_SRC * 4];
    int bstart[TTM_MAX_SRC * 4], bend[TTM_MAX_SRC * 4];
    int n = ttm_tokenize(&m, (const uint8_t *)text, (int)strlen(text), ids, bstart, bend, TTM_MAX_SRC * 4);
    if (n <= 0) { fprintf(stderr, "empty input\n"); ttm_unload(&m); return 1; }

    if (argc >= 4 && strcmp(argv[3], "--dump-tokens") == 0) {
        for (int i = 0; i < n; i++) printf("%u%c", ids[i], i + 1 < n ? ' ' : '\n');
        ttm_unload(&m);
        return 0;
    }

    // select (first 192 + salience top-64 of the tail), matching python
    int offs[TTM_MAX_SRC];
    if (n <= TTM_MAX_SRC) {
        for (int i = 0; i < n; i++) offs[i] = i;
    } else {
        int keep_first = 192;
        struct { float s; int i; } scored[TTM_MAX_SRC * 4];
        int nscored = 0;
        for (int i = keep_first; i < n; i++) {
            int l = bend[i] - bstart[i];
            const char *w = text + bstart[i];
            float s = 0.f;
            int has_digit = 0, cap = 0;
            if (l >= 1 && w[0] >= 'A' && w[0] <= 'Z') cap = 1;
            for (int c = 0; c < l; c++) if (w[c] >= '0' && w[c] <= '9') { has_digit = 1; break; }
            if (cap) s += 0.8f;
            if (has_digit) s += 0.5f;
            if (ids[i] == TTM_UNK_ID) s += 1.5f;
            s += 0.3f * (l < 8 ? l : 8);
            s += 0.02f * (i - keep_first);
            scored[nscored].s = s; scored[nscored].i = i;
            nscored++;
        }
        int m64 = nscored < 64 ? nscored : 64;
        for (int a = 0; a < m64; a++) {
            int best = a;
            for (int b = a + 1; b < nscored; b++)
                if (scored[b].s > scored[best].s) best = b;
            float ts = scored[a].s; int ti = scored[a].i;
            scored[a].s = scored[best].s; scored[a].i = scored[best].i;
            scored[best].s = ts; scored[best].i = ti;
        }
        int picked[64];
        for (int a = 0; a < m64; a++) picked[a] = scored[a].i;
        for (int a = 0; a < m64; a++)
            for (int b = a + 1; b < m64; b++)
                if (picked[b] < picked[a]) { int t = picked[a]; picked[a] = picked[b]; picked[b] = t; }
        for (int i = 0; i < keep_first; i++) offs[i] = i;
        for (int a = 0; a < m64; a++) offs[keep_first + a] = picked[a];
        n = keep_first + m64;
    }
    uint32_t sel_ids[TTM_MAX_SRC];
    int sel_bs[TTM_MAX_SRC], sel_be[TTM_MAX_SRC];
    for (int i = 0; i < n; i++) {
        sel_ids[i] = ids[offs[i]];
        sel_bs[i] = bstart[offs[i]];
        sel_be[i] = bend[offs[i]];
    }

    // Group selected source tokens into lexical words. Punctuation surrounding
    // a word is deliberately outside its copy span.
    int all_word_bs[TTM_MAX_SRC * 4], all_word_be[TTM_MAX_SRC * 4];
    int all_word_count = lexical_spans(text, all_word_bs, all_word_be, TTM_MAX_SRC * 4);
    int represented[TTM_MAX_SRC * 4];
    for (int i = 0; i < all_word_count; i++) represented[i] = -1;
    int token_word[TTM_MAX_SRC], word_bs[TTM_MAX_SRC], word_be[TTM_MAX_SRC];
    for (int i = 0; i < n; i++) token_word[i] = -1;
    int word_count = 0;
    for (int i = 0; i < n; i++) {
        for (int w = 0; w < all_word_count; w++) {
            if (sel_be[i] > all_word_bs[w] && sel_bs[i] < all_word_be[w]) {
                if (represented[w] < 0 && word_count < TTM_MAX_SRC) {
                    represented[w] = word_count;
                    word_bs[word_count] = all_word_bs[w];
                    word_be[word_count] = all_word_be[w];
                    word_count++;
                }
                token_word[i] = represented[w];
                break;
            }
        }
    }

    // workspace (static, bounded)
    size_t work_size = (size_t)n * D_EMB + 3 * D_ENC_HALF + 7 * D_ENC_HALF;
    float *work = malloc(work_size * sizeof(float));
    float *src_enc = malloc((size_t)n * D_ENC * sizeof(float));
    float *attn = malloc((size_t)n * sizeof(float));
    float *cov = calloc((size_t)n, sizeof(float));
    size_t gate_in = D_DEC + D_ENC + D_EMB;
    size_t dsize = (size_t)n * D_ATT + D_ENC + 2 * D_EMB +
                   gate_in + 2 * D_ATT + gate_in + D_ENC + 64;
    float *dscratch = malloc(dsize * sizeof(float));
    float *logits = malloc(V_MAX * sizeof(float));
    float *probs = malloc(V_MAX * sizeof(float));
    if (!work || !src_enc || !attn || !cov || !dscratch || !logits || !probs) { perror("alloc"); return 1; }

    // encode the source
    encode(&m, sel_ids, n, src_enc, work);

    // decoder init from final forward + final backward encoder summaries.
    float h[D_DEC];
    {
        float summary[D_ENC], row[D_ENC], diw[D_DEC];
        memcpy(summary, src_enc + (size_t)(n - 1) * D_ENC,
               D_ENC_HALF * sizeof(float));
        memcpy(summary + D_ENC_HALF, src_enc + D_ENC_HALF,
               D_ENC_HALF * sizeof(float));
        const float *bias = ttm_vec(&m, T_DEC_INIT_B, NULL);
        matvec(&m, T_DEC_INIT, D_DEC, D_ENC, summary, diw, row);
        for (int i = 0; i < D_DEC; i++) h[i] = tanhf(diw[i] + bias[i]);
    }

    // bounded beam-2 decode. Each hypothesis owns only fixed recurrent state,
    // coverage, and a small output buffer.
    const int max_tokens = 16, eos_id = 1;
    int beam_width = 1;
    const char *beam_env = getenv("TTM_BEAM_WIDTH");
    if (beam_env && atoi(beam_env) == 2) beam_width = TTM_BEAM;
    beam_hyp beams[TTM_BEAM], candidates[TTM_BEAM * TTM_BEAM];
    memset(beams, 0, sizeof(beams));
    memcpy(beams[0].h, h, sizeof(h));
    ttm_row(&m, T_EMB, 0, beams[0].prev_emb);

    int beam_count = 1;

    for (int t = 0; t < max_tokens; t++) {
        int candidate_count = 0;
        for (int b = 0; b < beam_count; b++) {
            if (beams[b].done) {
                candidates[candidate_count++] = beams[b];
                continue;
            }
            float p_gen, step_ctx[D_ENC];
            decode_step(&m, beams[b].h, beams[b].prev_emb, src_enc, n,
                        beams[b].cov, attn, logits, &p_gen, step_ctx, dscratch);
            vocab_probs(probs, logits, p_gen, 1.0f, m.vocab_count);
            if (t < 2 || rendered_word_count(beams[b].title) < 2) probs[eos_id] = 0.f;
            probs[0] = 0.f;
            float word_probs[TTM_MAX_SRC] = {0};
            for (int i = 0; i < n; i++)
                if (token_word[i] >= 0) word_probs[token_word[i]] += (1.f - p_gen) * attn[i];

            int top[TTM_BEAM] = {0, 0};
            float top_p[TTM_BEAM] = {-1.f, -1.f};
            for (int action = 0; action < (int)m.vocab_count + word_count; action++) {
                float probability = action < (int)m.vocab_count ? probs[action] :
                                    word_probs[action - (int)m.vocab_count];
                for (int k = 0; k < beam_width; k++) {
                    if (probability > top_p[k]) {
                        for (int q = beam_width - 1; q > k; q--) {
                            top_p[q] = top_p[q - 1]; top[q] = top[q - 1];
                        }
                        top_p[k] = probability; top[k] = action;
                        break;
                    }
                }
            }

            for (int k = 0; k < beam_width; k++) {
                if (top_p[k] <= 0.f || candidate_count >= TTM_BEAM * TTM_BEAM) continue;
                beam_hyp child = beams[b];
                child.score += logf(top_p[k] + 1e-30f);
                child.tokens++;
                int action = top[k];
                if (action == eos_id) {
                    child.done = 1;
                    candidates[candidate_count++] = child;
                    continue;
                }

                if (action < (int)m.vocab_count) {
                    append_generated(&child, &m, (uint32_t)action);
                    ttm_row(&m, T_EMB, action, child.prev_emb);
                } else {
                    int group = action - (int)m.vocab_count;
                    append_copied_word(&child, text, word_bs[group], word_be[group]);
                    memset(child.prev_emb, 0, sizeof(child.prev_emb));
                    int group_tokens = 0;
                    float token_emb[D_EMB];
                    for (int i = 0; i < n; i++) if (token_word[i] == group) {
                        ttm_row(&m, T_EMB, (int)sel_ids[i], token_emb);
                        for (int j = 0; j < D_EMB; j++) child.prev_emb[j] += token_emb[j];
                        group_tokens++;
                    }
                    if (group_tokens > 0)
                        for (int j = 0; j < D_EMB; j++) child.prev_emb[j] /= (float)group_tokens;
                }
                float ctx_proj[D_EMB], dec_input[D_EMB], rowbuf[D_ENC];
                matvec(&m, T_DEC_CTX, D_EMB, D_ENC, step_ctx, ctx_proj, rowbuf);
                for (int i = 0; i < D_EMB; i++)
                    dec_input[i] = child.prev_emb[i] + tanhf(ctx_proj[i]);
                float h_new[D_DEC], gscratch[7 * D_DEC];
                gru_cell(&m, T_DEC_W, T_DEC_U, T_DEC_B, D_EMB, D_DEC,
                         dec_input, beams[b].h, h_new, gscratch);
                memcpy(child.h, h_new, sizeof(h_new));
                for (int i = 0; i < n; i++) child.cov[i] = beams[b].cov[i] + attn[i];
                candidates[candidate_count++] = child;
            }
        }
        if (candidate_count == 0) break;
        for (int i = 0; i < candidate_count; i++) {
            int best = i;
            for (int j = i + 1; j < candidate_count; j++)
                if (beam_rank(&candidates[j]) > beam_rank(&candidates[best])) best = j;
            if (best != i) { beam_hyp tmp = candidates[i]; candidates[i] = candidates[best]; candidates[best] = tmp; }
        }
        beam_count = candidate_count < beam_width ? candidate_count : beam_width;
        for (int i = 0; i < beam_count; i++) beams[i] = candidates[i];
        int all_done = 1;
        for (int i = 0; i < beam_count; i++) if (!beams[i].done) all_done = 0;
        if (all_done) break;
    }
    int winner = 0;
    for (int i = 1; i < beam_count; i++)
        if (beam_rank(&beams[i]) > beam_rank(&beams[winner])) winner = i;
    beams[winner].title[beams[winner].title_len] = 0;
    collapse_duplicate_words(beams[winner].title);
    title_case_ascii(beams[winner].title);
    printf("%s\n", beams[winner].title);
    if (getenv("TTM_RSS")) {
        // read VmHWM from /proc/self/status (the plan's protocol; ru_maxrss is
        // unreliable on this system)
        FILE *pf = fopen("/proc/self/status", "r");
        char line[256];
        while (pf && fgets(line, sizeof(line), pf)) {
            if (strncmp(line, "VmHWM:", 6) == 0) {
                fprintf(stderr, "[rss] peak=%s kb\n", line + 6);
                break;
            }
        }
        if (pf) fclose(pf);
    }

    free(work); free(src_enc); free(attn); free(cov); free(dscratch); free(logits); free(probs);
    ttm_unload(&m);
    return 0;
}
