// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#include "layer.h"
#define PRINTF_DISABLE_SUPPORT_LONG_LONG
#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_redmule.h"
#include "flex_transpose_engine.h"
#include "flex_printf.h"
#include <math.h>
#include "vector.h"

#define NC ARCH_NUM_CLUSTER
#define HBUF(a) ((half_t *)(a))
#define FBUF(a) ((float *)(a))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define CEIL(a,b) (((a) + (b) - 1) / (b))

typedef struct { uint32_t expert[L_TOPK]; float weight[L_TOPK]; } Route;

static uint64_t channel(unsigned cid, unsigned off)
{
    return (uint64_t)ARCH_DRAM3D_ADDR_BASE + (uint64_t)cid * ARCH_DRAM3D_NODE_SPACE + off;
}

static uint64_t token(unsigned t, unsigned off, unsigned width)
{
    return channel(t % NC, off + t / NC * width * 2);
}

static uint64_t cache(unsigned job, unsigned off, unsigned pos)
{
    return channel(job % NC, off + job / NC * L_CACHE_STRIDE + pos * L_D * 2);
}

static void copy(uint64_t dst, uint64_t src, unsigned bytes)
{
    /* Serial phases drain before changing source backends on the legacy DMA. */
    __asm__ volatile("fence rw, rw" ::: "memory");
    flex_dma_async_1d(dst, src, bytes);
    flex_dma_async_wait_all();
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static void clear(unsigned dst, unsigned bytes)
{
    while (bytes) {
        unsigned part = MIN(bytes, ARCH_CLUSTER_ZOMEM_SIZE);
        copy(dst, zomem(0), part);
        bytes -= part; dst += part;
    }
}

static unsigned cycles(void)
{
    unsigned value;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
}

static float absf(float value) { return value < 0 ? -value : value; }

static void multiply(unsigned m, unsigned k, unsigned n, unsigned x, unsigned w, unsigned c)
{
    flex_redmule_config(m, k, n);
    flex_redmule_trigger(x, w, c, REDMULE_FP_16);
    flex_redmule_wait();
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static void add_half(half_t *dst, const half_t *a, const half_t *b, unsigned n, float *scratch)
{
    v_widen(scratch, a, n); v_widen(scratch + n, b, n);
    v_axpby(scratch, scratch, 1.0f, scratch + n, n);
    v_narrow(dst, scratch, n);
}

/* W is packed [local output tile][K tile][KT][NT]. Dense output tiles are
 * cyclic across channels, whereas every tile of an expert stays at its owner.
 * Row addresses permit gathering routed tokens and scattering top-k results.
 */
static void linear(unsigned cid, unsigned m, unsigned k, unsigned n,
                   const uint64_t *input, const uint64_t *output,
                   unsigned woff, unsigned boff, unsigned distributed)
{
    unsigned kp = CEIL(k, L_KT) * L_KT;
    clear(L_L1_A, m * kp * 2);
    for (unsigned r = 0; r < m; ++r)
        copy(L_L1_A + r * kp * 2, input[r], k * 2);
    for (unsigned j = distributed ? cid : 0; j < CEIL(n, L_NT); j += distributed ? NC : 1) {
        unsigned local = distributed ? j / NC : j;
        unsigned valid = MIN(L_NT, n - j * L_NT);
        clear(L_L1_C, m * L_NT * 2);
        for (unsigned kb = 0; kb < kp / L_KT; ++kb) {
            flex_dma_async_2d(L_L1_X, L_L1_A + kb * L_KT * 2,
                              L_KT * 2, L_KT * 2, kp * 2, m);
            flex_dma_async_wait_all();
            copy(L_L1_W, channel(cid, woff + (local * (kp / L_KT) + kb) * L_KT * L_NT * 2),
                 L_KT * L_NT * 2);
            multiply(m, L_KT, L_NT, L_L1_X, L_L1_W, L_L1_C);
        }
        copy(L_L1_BIAS, channel(cid, boff + local * L_NT * 2), L_NT * 2);
        for (unsigned r = 0; r < m; ++r) {
            add_half(HBUF(L_L1_C) + r * L_NT, HBUF(L_L1_C) + r * L_NT,
                     HBUF(L_L1_BIAS), valid, FBUF(L_L1_TMP));
            copy(output[r] + j * L_NT * 2, L_L1_C + r * L_NT * 2, valid * 2);
        }
    }
}

static void projection(unsigned cid, unsigned inoff, unsigned outoff,
                       unsigned k, unsigned n, unsigned woff, unsigned boff)
{
    for (unsigned first = 0; first < L_TOKENS; first += L_MT) {
        unsigned m = MIN(L_MT, L_TOKENS - first);
        uint64_t in[L_MT], out[L_MT];
        for (unsigned r = 0; r < m; ++r) {
            in[r] = token(first + r, inoff, k);
            out[r] = token(first + r, outoff, n);
        }
        linear(cid, m, k, n, in, out, woff, boff, 1);
    }
}

static void norm(unsigned cid, unsigned inoff, unsigned outoff, unsigned weightoff)
{
    unsigned weights = L_L1_TMP + L_H * 2;
    copy(weights, channel(cid, weightoff), L_H * 2);
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        copy(L_L1_TMP, token(t, inoff, L_H), L_H * 2);
        v_rmsnorm(HBUF(L_L1_TMP), HBUF(L_L1_TMP), HBUF(weights), L_H, FBUF(L_L1_A));
        copy(token(t, outoff, L_H), L_L1_TMP, L_H * 2);
    }
}

static void rope_append(unsigned cid)
{
    unsigned table = L_L1_TMP + L_QKV * 2;
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        copy(L_L1_TMP, token(t, L_OFF_QKV, L_QKV), L_QKV * 2);
        copy(table, channel(cid, L_OFF_ROPE + t * L_D * 2), L_D * 2);
        for (unsigned h = 0; h < L_HEADS + L_KV_HEADS; ++h)
            v_rope(HBUF(L_L1_TMP) + h * L_D, HBUF(table), L_D, FBUF(L_L1_A));
        copy(token(t, L_OFF_QKV, L_QKV), L_L1_TMP, L_QKV * 2);
        unsigned b = t / L_SEQ, pos = L_PAST + t % L_SEQ;
        for (unsigned h = 0; h < L_KV_HEADS; ++h) {
            unsigned job = b * L_KV_HEADS + h;
            copy(cache(job, L_OFF_CACHE_K, pos), L_L1_TMP + (L_QDIM + h * L_D) * 2, L_D * 2);
            copy(cache(job, L_OFF_CACHE_V, pos), L_L1_TMP + (L_QDIM + L_KVDIM + h * L_D) * 2, L_D * 2);
        }
    }
}

static void attention(unsigned cid)
{
    unsigned q = L_L1_A, k = q + L_GROUP * L_D * 2;
    unsigned kt = k + L_BC * L_D * 2, v = kt + L_BC * L_D * 2;
    unsigned p = v + L_BC * L_D * 2, pv = p + L_GROUP * L_BC * 2;
    float *acc = FBUF(pv + L_GROUP * L_D * 2);
    float *maximum = acc + L_GROUP * L_D;
    float *denominator = maximum + L_GROUP, *alpha = denominator + L_GROUP;
    float *scores = FBUF(L_L1_TMP);
    for (unsigned job = cid; job < L_BATCH * L_KV_HEADS; job += NC) {
        unsigned b = job / L_KV_HEADS, head = job % L_KV_HEADS;
        for (unsigned t = 0; t < L_SEQ; ++t) {
            unsigned token_id = b * L_SEQ + t, end = L_PAST + t + 1;
            unsigned first = L_WINDOW && end > L_WINDOW ? end - L_WINDOW : 0;
            copy(q, token(token_id, L_OFF_QKV, L_QKV) + head * L_GROUP * L_D * 2, L_GROUP * L_D * 2);
            copy(L_L1_BIAS, channel(cid, L_OFF_SINKS + head * L_GROUP * 2), L_GROUP * 2);
            v_widen(maximum, HBUF(L_L1_BIAS), L_GROUP);
            clear((unsigned)acc, L_GROUP * L_D * 4);
            for (unsigned h = 0; h < L_GROUP; ++h) denominator[h] = 1.0f;
            for (unsigned start = first; start < end; start += L_BC) {
                unsigned valid = MIN(L_BC, end - start);
                clear(k, L_BC * L_D * 2); clear(v, L_BC * L_D * 2);
                copy(k, cache(job, L_OFF_CACHE_K, start), valid * L_D * 2);
                copy(v, cache(job, L_OFF_CACHE_V, start), valid * L_D * 2);
                flex_transpose_engine_config(L_BC, L_D, k, kt, 2);
                flex_transpose_engine_trigger(); flex_transpose_engine_wait();
                clear(p, L_GROUP * L_BC * 2);
                multiply(L_GROUP, L_D, L_BC, q, kt, p);
                for (unsigned h = 0; h < L_GROUP; ++h) {
                    v_widen(scores, HBUF(p) + h * L_BC, valid);
                    v_scale(scores, scores, L_SM_SCALE, valid);
                    float next = v_reduce(scores, valid, maximum[h], 1);
                    alpha[h] = maximum[h] - next;
                    maximum[h] = next;
                    v_exp(scores, scores, next, valid);
                    float sum = v_reduce(scores, valid, 0.0f, 0);
                    v_narrow(HBUF(p) + h * L_BC, scores, valid);
                    v_exp(alpha + h, alpha + h, 0.0f, 1);
                    denominator[h] = denominator[h] * alpha[h] + sum;
                }
                clear(pv, L_GROUP * L_D * 2);
                multiply(L_GROUP, L_BC, L_D, p, v, pv);
                for (unsigned h = 0; h < L_GROUP; ++h) {
                    v_widen(scores, HBUF(pv) + h * L_D, L_D);
                    v_axpby(acc + h * L_D, acc + h * L_D, alpha[h], scores, L_D);
                }
            }
            for (unsigned h = 0; h < L_GROUP; ++h) {
                v_scale(scores, acc + h * L_D, 1.0f / denominator[h], L_D);
                v_narrow(HBUF(pv) + h * L_D, scores, L_D);
            }
            copy(token(token_id, L_OFF_ATTN, L_QDIM) + head * L_GROUP * L_D * 2,
                 pv, L_GROUP * L_D * 2);
        }
    }
}

static void residual(unsigned cid)
{
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        copy(L_L1_TMP, token(t, L_OFF_X, L_H), L_H * 2);
        copy(L_L1_TMP + L_H * 2, token(t, L_OFF_PROJECTED, L_H), L_H * 2);
        add_half(HBUF(L_L1_TMP), HBUF(L_L1_TMP), HBUF(L_L1_TMP) + L_H,
                 L_H, FBUF(L_L1_A));
        copy(token(t, L_OFF_RESIDUAL, L_H), L_L1_TMP, L_H * 2);
    }
}

static void route(unsigned cid)
{
    float *logits = FBUF(L_L1_A), *selected = logits + L_EXPERTS;
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        Route *r = (Route *)L_L1_ROUTES;
        copy(L_L1_TMP, token(t, L_OFF_ROUTER, L_EXPERTS), L_EXPERTS * 2);
        v_widen(logits, HBUF(L_L1_TMP), L_EXPERTS);
        for (unsigned slot = 0; slot < L_TOPK; ++slot) {
            float best = v_reduce(logits, L_EXPERTS, -INFINITY, 1);
            unsigned e = 0;
            /* This Spatz ISA lacks vfirst.m. Scan only for the index of the
               vector maximum; smaller indices win ties, as in the reference. */
            while (e < L_EXPERTS && logits[e] != best) ++e;
            if (e == L_EXPERTS || !(loaded_experts[e / 32] & (1u << (e % 32)))) {
                printf("LLM_ROUTING_FAIL token=%u expert=%u\n", t, e);
                flex_eoc(1);
                while (1) __asm__ volatile("wfi");
            }
            r->expert[slot] = e;
            selected[slot] = best;
            logits[e] = -INFINITY;
        }
        v_exp(selected, selected, selected[0], L_TOPK);
        float sum = v_reduce(selected, L_TOPK, 0.0f, 0);
        v_scale(r->weight, selected, 1.0f / sum, L_TOPK);
        copy(channel(cid, L_OFF_ROUTES + t / NC * sizeof(Route)), L_L1_ROUTES, sizeof(Route));
    }
}

static void experts(unsigned cid)
{
    Route *routes = (Route *)L_L1_ROUTES;
    for (unsigned t = 0; t < L_TOKENS; ++t)
        copy((unsigned)&routes[t], channel(t % NC, L_OFF_ROUTES + t / NC * sizeof(Route)), sizeof(Route));
    for (unsigned e = cid; e < L_EXPERTS; e += NC) {
        unsigned tokens[L_TOKENS], slots[L_TOKENS], count = 0;
        for (unsigned t = 0; t < L_TOKENS; ++t)
            for (unsigned slot = 0; slot < L_TOPK; ++slot)
                if (routes[t].expert[slot] == e) { tokens[count] = t; slots[count++] = slot; }
        unsigned extra = e / NC * L_EXPERT_STRIDE;
        for (unsigned first = 0; first < count; first += L_MT) {
            unsigned m = MIN(L_MT, count - first);
            uint64_t in[L_MT], out[L_MT];
            for (unsigned r = 0; r < m; ++r) {
                in[r] = token(tokens[first + r], L_OFF_NORM2, L_H);
                out[r] = channel(cid, L_OFF_UP + r * 2 * L_I * 2);
            }
            linear(cid, m, L_H, 2 * L_I, in, out, L_OFF_W_UP + extra, L_OFF_B_UP + extra, 0);
            for (unsigned r = 0; r < m; ++r) {
                copy(L_L1_A, out[r], 2 * L_I * 2);
                v_swiglu(HBUF(L_L1_TMP), HBUF(L_L1_A), L_I);
                in[r] = channel(cid, L_OFF_ACTIVATION + r * L_I * 2);
                copy(in[r], L_L1_TMP, L_I * 2);
                out[r] = token(tokens[first + r], L_OFF_PARTIAL, L_TOPK * L_H) + slots[first + r] * L_H * 2;
            }
            linear(cid, m, L_I, L_H, in, out, L_OFF_W_DOWN + extra, L_OFF_B_DOWN + extra, 0);
        }
    }
}

static void combine(unsigned cid)
{
    float *acc = FBUF(L_L1_A), *value = acc + L_H;
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        Route *r = (Route *)L_L1_ROUTES;
        copy(L_L1_ROUTES, channel(cid, L_OFF_ROUTES + t / NC * sizeof(Route)), sizeof(Route));
        clear(L_L1_A, L_H * 4);
        for (unsigned slot = 0; slot < L_TOPK; ++slot) {
            copy(L_L1_TMP, token(t, L_OFF_PARTIAL, L_TOPK * L_H) + slot * L_H * 2, L_H * 2);
            v_widen(value, HBUF(L_L1_TMP), L_H);
            v_axpby(acc, value, r->weight[slot], acc, L_H);
        }
        copy(L_L1_TMP, token(t, L_OFF_RESIDUAL, L_H), L_H * 2);
        v_widen(value, HBUF(L_L1_TMP), L_H);
        v_axpby(acc, acc, 1.0f, value, L_H);
        v_narrow(HBUF(L_L1_TMP), acc, L_H);
        copy(token(t, L_OFF_OUTPUT, L_H), L_L1_TMP, L_H * 2);
    }
}

/* Validation is outside the timed layer. Check intermediate values so a
 * residual connection cannot hide a broken attention or expert sub-block. */
static unsigned check_half(uint64_t actual, uint64_t expected, unsigned n,
                           unsigned cid, unsigned field, unsigned *checked)
{
    unsigned errors = 0;
    for (unsigned j = 0; j < n; j += 512) {
        unsigned size = MIN(512, n - j);
        copy(L_L1_A, actual + j * 2, size * 2);
        copy(L_L1_W, expected + j * 2, size * 2);
        for (unsigned i = 0; i < size; ++i) {
            float a = HBUF(L_L1_A)[i], b = HBUF(L_L1_W)[i];
            if (!isfinite(a) || !isfinite(b) || absf(a - b) > 0.003f + 0.02f * absf(b)) {
                if (!errors) printf("LLM_MISMATCH cid=%u field=%u index=%u actual=%f expected=%f\n",
                                    cid, field, j + i, (double)a, (double)b);
                ++errors;
            }
        }
        *checked += size;
    }
    return errors;
}

static unsigned check(unsigned cid, unsigned *checked)
{
    static const unsigned fields[][3] = {
        {L_OFF_NORM1, L_OFF_GOLD_NORM1, L_H}, {L_OFF_QKV, L_OFF_GOLD_QKV, L_QKV},
        {L_OFF_ATTN, L_OFF_GOLD_ATTN, L_QDIM}, {L_OFF_PROJECTED, L_OFF_GOLD_PROJECTED, L_H},
        {L_OFF_RESIDUAL, L_OFF_GOLD_RESIDUAL, L_H}, {L_OFF_NORM2, L_OFF_GOLD_NORM2, L_H},
        {L_OFF_ROUTER, L_OFF_GOLD_ROUTER, L_EXPERTS}, {L_OFF_PARTIAL, L_OFF_GOLD_PARTIAL, L_TOPK * L_H},
        {L_OFF_OUTPUT, L_OFF_GOLD_OUTPUT, L_H}};
    unsigned errors = 0;
    for (unsigned t = cid; t < L_TOKENS; t += NC) {
        for (unsigned f = 0; f < sizeof(fields) / sizeof(fields[0]); ++f)
            errors += check_half(token(t, fields[f][0], fields[f][2]),
                                 token(t, fields[f][1], fields[f][2]), fields[f][2], cid, f, checked);
        for (unsigned h = 0; h < L_KV_HEADS; ++h) {
            unsigned job = t / L_SEQ * L_KV_HEADS + h;
            uint64_t gold = token(t, L_OFF_GOLD_QKV, L_QKV) + (L_QDIM + h * L_D) * 2;
            errors += check_half(cache(job, L_OFF_CACHE_K, L_PAST + t % L_SEQ), gold, L_D, cid, 9, checked);
            errors += check_half(cache(job, L_OFF_CACHE_V, L_PAST + t % L_SEQ), gold + L_KVDIM * 2, L_D, cid, 10, checked);
        }
        copy(L_L1_A, channel(cid, L_OFF_ROUTES + t / NC * sizeof(Route)), sizeof(Route));
        copy(L_L1_W, channel(cid, L_OFF_GOLD_ROUTES + t / NC * sizeof(Route)), sizeof(Route));
        Route *a = (Route *)L_L1_A, *b = (Route *)L_L1_W;
        for (unsigned s = 0; s < L_TOPK; ++s) {
            unsigned bad = a->expert[s] != b->expert[s] || !isfinite(a->weight[s]) ||
                           absf(a->weight[s] - b->weight[s]) > 0.002f;
            if (bad) printf("LLM_ROUTE_MISMATCH token=%u slot=%u actual=%u expected=%u\n",
                            t, s, a->expert[s], b->expert[s]);
            errors += bad;
            *checked += 2;
        }
    }
    return errors;
}

int main(void)
{
    unsigned cid = flex_get_cluster_id(), first = flex_is_first_core();
    unsigned previous = 0, stages[11];
    uint64_t elapsed = 0;
    flex_barrier_xy_init(); flex_global_barrier_xy();
    if (cid == 0 && first) { flex_timer_start(); previous = cycles(); }
    flex_global_barrier_xy();
#define STAGE(id, operation) do { \
    if (first) { operation; } \
    flex_global_barrier_xy(); \
    if (cid == 0 && first) { unsigned now = cycles(); stages[id] = now - previous; \
        elapsed += stages[id]; previous = now; } \
} while (0)
    STAGE(0, norm(cid, L_OFF_X, L_OFF_NORM1, L_OFF_NORM1_WEIGHT));
    STAGE(1, projection(cid, L_OFF_NORM1, L_OFF_QKV, L_H, L_QKV, L_OFF_W_QKV, L_OFF_B_QKV));
    STAGE(2, rope_append(cid));
    STAGE(3, attention(cid));
    STAGE(4, projection(cid, L_OFF_ATTN, L_OFF_PROJECTED, L_QDIM, L_H, L_OFF_W_OUT, L_OFF_B_OUT));
    STAGE(5, residual(cid));
    STAGE(6, norm(cid, L_OFF_RESIDUAL, L_OFF_NORM2, L_OFF_NORM2_WEIGHT));
    STAGE(7, projection(cid, L_OFF_NORM2, L_OFF_ROUTER, L_H, L_EXPERTS, L_OFF_W_ROUTER, L_OFF_B_ROUTER));
    STAGE(8, route(cid));
    STAGE(9, experts(cid));
    STAGE(10, combine(cid));
    if (cid == 0 && first) flex_timer_end();
    flex_global_barrier_xy();
    unsigned checked = 0, errors = first ? check(cid, &checked) : 0;
    if (first) {
        *(volatile unsigned *)(ARCH_SYNC_BASE + cid * ARCH_SYNC_SIZE + 32) = errors;
        *(volatile unsigned *)(ARCH_SYNC_BASE + cid * ARCH_SYNC_SIZE + 36) = checked;
    }
    flex_global_barrier_xy();
    if (cid == 0 && first) {
        checked = errors = 0;
        for (unsigned c = 0; c < NC; ++c) {
            errors += *(volatile unsigned *)(ARCH_SYNC_BASE + c * ARCH_SYNC_SIZE + 32);
            checked += *(volatile unsigned *)(ARCH_SYNC_BASE + c * ARCH_SYNC_SIZE + 36);
        }
        for (unsigned i = 0; i < 11; ++i)
            printf("LLM_STAGE id=%u cycles=%u\n", i, stages[i]);
        printf("LLM_DRAM3D_%s batch=%u seq=%u past=%u layer=%u cycles_hi=%u cycles_lo=%u errors=%u checked=%u\n",
               errors ? "FAIL" : "PASS", L_BATCH, L_SEQ, L_PAST, L_LAYER,
               (unsigned)(elapsed >> 32), (unsigned)elapsed, errors, checked);
        flex_eoc(errors != 0);
    }
    return 0;
}
