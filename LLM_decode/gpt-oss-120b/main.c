// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
// One complete GPT-OSS decoder layer. Scheduling follows the SDK's DMA/RedMule
// kernels; tensor storage is FP16, with FP32 scalar reductions and normalization.
#include <math.h>
#include <stdint.h>
#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_redmule.h"
#include "flex_dump.h"
#include "model.h"
#include "vector.h"

#define H MODEL_HIDDEN_SIZE
#define F MODEL_INTERMEDIATE_SIZE
#define E MODEL_NUM_EXPERTS
#define TOP MODEL_EXPERTS_PER_TOKEN
#define TK MODEL_TILE_K
#define TN MODEL_TILE_N
#define AT MODEL_ATTENTION_TILE
#define D MODEL_HEAD_DIM
#define X0 local(0x01000)
#define X1 local(0x09000)
#define W0 local(0x11000)
#define W1 local(0x21000)
#define Z0 local(0x31000)
#define BIAS local(0x35000)
#define META local(0x5c000)

typedef struct {
    uint32_t count[E], offset[E + 1];
    uint32_t token[MODEL_MAX_BATCH * TOP], rank[MODEL_MAX_BATCH * TOP];
} RouteMeta;

_Static_assert(sizeof(RouteMeta) <= 0x4000, "Route metadata exceeds TCDM reservation");
_Static_assert(MODEL_MAX_BATCH * TK * 2 <= X1 - X0, "Input tile too large");
_Static_assert(TK * TN * 2 <= W1 - W0, "Weight tile too large");
_Static_assert(MODEL_MAX_BATCH * TN * 2 <= BIAS - Z0, "Output tile too large");
_Static_assert(H * 4 <= W1 - W0, "Normalization scratch too large");

static unsigned batch, context, dump_detail, cid;
#define core (flex_get_core_id())
static uint64_t ticks[14];
static uint32_t previous;

static uint32_t cycle(void)
{
    // SoftHier exposes cycle, but not cycleh. Accumulate stage deltas in 64 bits.
    uint32_t value;
    asm volatile("rdcycle %0" : "=r"(value));
    return value;
}

static void number(uint64_t value)
{
    char chars[24]; unsigned n = 0;
    // Avoid multilib helpers compiled with RVC: this core has no C extension.
    do {
        unsigned remainder = 0;
        uint64_t quotient = 0;
        for (unsigned bit = 0; bit < 64; ++bit) {
            remainder = (remainder << 1) | (unsigned)(value >> 63);
            value <<= 1; quotient <<= 1;
            if (remainder >= 10) { remainder -= 10; quotient |= 1; }
        }
        chars[n++] = '0' + remainder; value = quotient;
    } while (value);
    while (n) flex_log_char(chars[--n]);
}

static inline float scalar_sqrt(float value)
{
    float result;
    asm ("fsqrt.s %0, %1" : "=f"(result) : "f"(value));
    return result;
}

static inline float scalar_min(float a, float b) { return a < b ? a : b; }
static inline float scalar_max(float a, float b) { return a > b ? a : b; }

static float scalar_exp(float value)
{
    // Vector loads/stores use TCDM; reserve a scalar scratch word outside tiles.
    float *scratch = (float *)local(0x38000);
    *scratch = value;
    vec_exp32(scratch, 1);
    return *scratch;
}

static void copy(uint64_t dst, uint64_t src, unsigned bytes)
{
    asm volatile("" ::: "memory");
    flex_dma_async_1d(dst, src, bytes);
    flex_dma_async_wait_all();
    asm volatile("" ::: "memory");
}

static void copy2(uint64_t dst, uint64_t src, unsigned width,
                  unsigned dst_stride, unsigned src_stride, unsigned rows)
{
    asm volatile("" ::: "memory");
    flex_dma_async_2d(dst, src, width, dst_stride, src_stride, rows);
    flex_dma_async_wait_all();
    asm volatile("" ::: "memory");
}

static void phase(unsigned index)
{
    flex_global_barrier_xy();
    if (cid == 0 && core == 0) { uint32_t now = cycle(); ticks[index] = (uint32_t)(now - previous); previous = now; }
}

static void norm(uint64_t output, uint64_t input, uint64_t gamma)
{
    if (core) return;
    half *x = (half *)X0, *scale = (half *)X1, *out = (half *)Z0;
    copy(X1, gamma, H * 2);
    for (unsigned row = cid; row < batch; row += ARCH_NUM_CLUSTER) {
        copy(X0, input + (uint64_t)row * H * 2, H * 2);
        float sum = 0;
        for (unsigned j = 0; j < H; ++j) { float a = hfloat(x[j]); sum += a * a; }
        float inv = 1.0f / scalar_sqrt(sum / H + MODEL_RMS_NORM_EPS);
        for (unsigned j = 0; j < H; ++j) out[j] = fhalf(hfloat(x[j]) * inv * hfloat(scale[j]));
        copy(output + (uint64_t)row * H * 2, Z0, H * 2);
    }
}

static void load_tile(unsigned slot, uint64_t input, uint64_t weight,
                      unsigned rows, unsigned k, unsigned kb)
{
    unsigned x = slot ? X1 : X0, w = slot ? W1 : W0;
    unsigned valid = k - kb * TK; if (valid > TK) valid = TK;
    if (valid != TK) copy(x, zomem(0), rows * TK * 2);
    flex_dma_async_2d(x, input + (uint64_t)kb * TK * 2, valid * 2, TK * 2, k * 2, rows);
    flex_dma_async_1d(w, weight + (uint64_t)kb * TK * TN * 2, TK * TN * 2);
    flex_dma_async_wait_all();
    asm volatile("" ::: "memory");
}

// Output-column tiling supports small batches and dimensions with partial tiles.
// Weights are packed [N-tile, K-tile, TK, TN], with zero padding outside the matrix.
static void linear(uint64_t output, uint64_t input, uint64_t weight, uint64_t bias,
                   unsigned m, unsigned k, unsigned n)
{
    if (core || !m) return;
    unsigned nk = (k + TK - 1) / TK, nn = (n + TN - 1) / TN;
    for (unsigned tile = cid; tile < nn; tile += ARCH_NUM_CLUSTER) {
        unsigned valid = n - tile * TN; if (valid > TN) valid = TN;
        uint64_t packed = weight + (uint64_t)tile * nk * TK * TN * 2;
        copy(Z0, zomem(0), m * TN * 2);
        load_tile(0, input, packed, m, k, 0);
        for (unsigned kb = 0; kb < nk; ++kb) {
            flex_redmule_config(m, TK, TN);
            flex_redmule_trigger(kb & 1 ? X1 : X0, kb & 1 ? W1 : W0, Z0, REDMULE_FP_16);
            if (kb + 1 < nk) load_tile((kb + 1) & 1, input, packed, m, k, kb + 1);
            flex_redmule_wait();
        }
        if (bias) {
            copy(BIAS, bias + tile * TN * 2, valid * 2);
            for (unsigned row = 0; row < m; ++row)
                vec_add((half *)Z0 + row * TN, (half *)Z0 + row * TN, (half *)BIAS, valid);
        }
        copy2(output + (uint64_t)tile * TN * 2, Z0, valid * 2, n * 2, TN * 2, m);
    }
}

static void rotate_and_cache(void)
{
    if (core) return;
    half *qkv = (half *)X0, *cs = (half *)X1;
    copy(X1, HBM_ROPE, D * 2);
    for (unsigned row = cid; row < batch; row += ARCH_NUM_CLUSTER) {
        copy(X0, HBM_QKV + (uint64_t)row * QKV_SIZE * 2, QKV_SIZE * 2);
        for (unsigned head = 0; head < MODEL_NUM_ATTENTION_HEADS + MODEL_NUM_KEY_VALUE_HEADS; ++head) {
            half *x = qkv + head * D;
            for (unsigned j = 0; j < D / 2; ++j) {
                float a = hfloat(x[j]), z = hfloat(x[j + D / 2]);
                float c = hfloat(cs[j]), s = hfloat(cs[j + D / 2]);
                x[j] = fhalf(a * c - z * s); x[j + D / 2] = fhalf(z * c + a * s);
            }
        }
        copy(HBM_QKV + (uint64_t)row * QKV_SIZE * 2, X0, QKV_SIZE * 2);
        for (unsigned head = 0; head < MODEL_NUM_KEY_VALUE_HEADS; ++head) {
            uint64_t index = (uint64_t)row * MODEL_NUM_KEY_VALUE_HEADS + head;
            copy2(HBM_KEY_CACHE + (index * D * CACHE_STRIDE + context) * 2,
                  X0 + (Q_SIZE + head * D) * 2, 2, CACHE_STRIDE * 2, 2, D);
            copy(HBM_VALUE_CACHE + (index * CACHE_STRIDE + context) * D * 2,
                 X0 + (Q_SIZE + KV_SIZE + head * D) * 2, D * 2);
        }
    }
}

static void attention(void)
{
    if (core) return;
    // One task owns one KV head and its Q_GROUP query heads. K is transposed in HBM,
    // V is row-major. This reuses KV tiles across query heads as in FlatAttention.
    unsigned q = local(0x1000), kt = local(0x2000), v = local(0x8000);
    unsigned scores = local(0x10000), tile = local(0x19000), out = local(0x1a000);
    unsigned p = local(0x1b000), scratch = local(0x20000), sinks = local(0x25000);
    float scale = 1.0f / scalar_sqrt((float)D);
    copy(sinks, HBM_SINKS, MODEL_NUM_ATTENTION_HEADS * 2);
    for (unsigned job = cid; job < batch * MODEL_NUM_KEY_VALUE_HEADS; job += ARCH_NUM_CLUSTER) {
        unsigned row = job / MODEL_NUM_KEY_VALUE_HEADS, head = job % MODEL_NUM_KEY_VALUE_HEADS;
        copy(q, HBM_QKV + ((uint64_t)row * QKV_SIZE + head * Q_GROUP * D) * 2, Q_GROUP * D * 2);
        copy(scores, zomem(0), Q_GROUP * CACHE_STRIDE * 2);
        for (unsigned pos = 0; pos <= context; pos += AT) {
            copy2(kt, HBM_KEY_CACHE + ((uint64_t)job * D * CACHE_STRIDE + pos) * 2,
                  AT * 2, AT * 2, CACHE_STRIDE * 2, D);
            copy(tile, zomem(0), Q_GROUP * AT * 2);
            flex_redmule_config(Q_GROUP, D, AT);
            flex_redmule_trigger(q, kt, tile, REDMULE_FP_16); flex_redmule_wait();
            copy2(scores + pos * 2, tile, AT * 2, CACHE_STRIDE * 2, AT * 2, Q_GROUP);
        }
        for (unsigned group = 0; group < Q_GROUP; ++group) {
            half *s = (half *)scores + group * CACHE_STRIDE;
            float *tmp = (float *)scratch;
            float sink = hfloat(((half *)sinks)[head * Q_GROUP + group]), max = sink;
            for (unsigned j = 0; j <= context; ++j) {
                tmp[j] = hfloat(s[j]) * scale;
                if (tmp[j] > max) max = tmp[j];
            }
            for (unsigned j = 0; j <= context; ++j) tmp[j] -= max;
            vec_exp32(tmp, context + 1);
            float sum = scalar_exp(sink - max);
            for (unsigned j = 0; j <= context; ++j) sum += tmp[j];
            float inv = 1.0f / sum;
            for (unsigned j = 0; j <= context; ++j) s[j] = fhalf(tmp[j] * inv);
            // Padded positions are masked; the sink contributes only to the denominator.
            for (unsigned j = context + 1; j < CACHE_STRIDE; ++j) s[j] = 0;
        }
        copy(out, zomem(0), Q_GROUP * D * 2);
        for (unsigned pos = 0; pos <= context; pos += AT) {
            copy2(p, scores + pos * 2, AT * 2, AT * 2, CACHE_STRIDE * 2, Q_GROUP);
            copy(v, HBM_VALUE_CACHE + ((uint64_t)job * CACHE_STRIDE + pos) * D * 2, AT * D * 2);
            flex_redmule_config(Q_GROUP, AT, D);
            flex_redmule_trigger(p, v, out, REDMULE_FP_16); flex_redmule_wait();
        }
        copy(HBM_ATTENTION + ((uint64_t)row * Q_SIZE + head * Q_GROUP * D) * 2,
             out, Q_GROUP * D * 2);
    }
}

static void residual(void)
{
    if (core) return;
    for (unsigned row = cid; row < batch; row += ARCH_NUM_CLUSTER) {
        copy(X0, HBM_INPUT + (uint64_t)row * H * 2, H * 2);
        copy(X1, HBM_PROJECTION + (uint64_t)row * H * 2, H * 2);
        vec_add((half *)Z0, (half *)X0, (half *)X1, H);
        copy(HBM_RESIDUAL + (uint64_t)row * H * 2, Z0, H * 2);
    }
}

static void route(void)
{
    if (core || cid) return;
    RouteMeta *meta = (RouteMeta *)META;
    unsigned *ids = (unsigned *)W0, *cursor = (unsigned *)W1;
    half *probs = (half *)Z0, *logits = (half *)X0;
    for (unsigned e = 0; e < E; ++e) meta->count[e] = 0;
    for (unsigned b = 0; b < batch; ++b) {
        copy(X0, HBM_ROUTER_LOGITS + (uint64_t)b * E * 2, E * 2);
        float selected[TOP];
        for (unsigned rank = 0; rank < TOP; ++rank) {
            float max = -INFINITY; unsigned expert = 0;
            for (unsigned e = 0; e < E; ++e) if (hfloat(logits[e]) > max) { max = hfloat(logits[e]); expert = e; }
            ids[b * TOP + rank] = expert; selected[rank] = max;
            logits[expert] = 0xfc00; ++meta->count[expert];
        }
        float sum = 0, max = selected[0];
        for (unsigned rank = 0; rank < TOP; ++rank) { selected[rank] = scalar_exp(selected[rank] - max); sum += selected[rank]; }
        for (unsigned rank = 0; rank < TOP; ++rank) probs[b * TOP + rank] = fhalf(selected[rank] / sum);
    }
    meta->offset[0] = 0;
    for (unsigned e = 0; e < E; ++e) { meta->offset[e + 1] = meta->offset[e] + meta->count[e]; cursor[e] = meta->offset[e]; }
    for (unsigned b = 0; b < batch; ++b) for (unsigned rank = 0; rank < TOP; ++rank) {
        unsigned row = cursor[ids[b * TOP + rank]]++;
        meta->token[row] = b; meta->rank[row] = rank;
    }
    copy(HBM_ROUTES, W0, batch * TOP * 4);
    copy(HBM_ROUTE_WEIGHTS, Z0, batch * TOP * 2);
    copy(HBM_METADATA, META, sizeof(*meta));
}

static void dispatch(void)
{
    if (core) return;
    copy(META, HBM_METADATA, sizeof(RouteMeta));
    RouteMeta *meta = (RouteMeta *)META;
    for (unsigned row = cid; row < batch * TOP; row += ARCH_NUM_CLUSTER)
        copy(HBM_DISPATCH + (uint64_t)row * H * 2,
             HBM_NORM2_OUT + (uint64_t)meta->token[row] * H * 2, H * 2);
}

static void experts_up(void)
{
    if (core) return;
    RouteMeta *meta = (RouteMeta *)META;
    for (unsigned e = 0; e < E; ++e)
        linear(HBM_GATE_UP + (uint64_t)meta->offset[e] * F * 4,
               HBM_DISPATCH + (uint64_t)meta->offset[e] * H * 2,
               HBM_UP_WEIGHT[e], HBM_UP_BIAS[e], meta->count[e], H, F * 2);
}

static void swiglu(void)
{
    if (core) return;
    half *pairs = (half *)X0, *out = (half *)Z0;
    float *tmp = (float *)W0;
    for (unsigned row = cid; row < batch * TOP; row += ARCH_NUM_CLUSTER) {
        copy(X0, HBM_GATE_UP + (uint64_t)row * F * 4, F * 4);
        for (unsigned i = 0; i < F; ++i) tmp[i] = -MODEL_SWIGLU_ALPHA * scalar_min(hfloat(pairs[2 * i]), MODEL_SWIGLU_LIMIT);
        vec_exp32(tmp, F);
        for (unsigned i = 0; i < F; ++i) {
            float gate = scalar_min(hfloat(pairs[2 * i]), MODEL_SWIGLU_LIMIT);
            float up = scalar_min(scalar_max(hfloat(pairs[2 * i + 1]), -MODEL_SWIGLU_LIMIT), MODEL_SWIGLU_LIMIT);
            out[i] = fhalf(gate / (1 + tmp[i]) * (up + 1));
        }
        copy(HBM_ACTIVATION + (uint64_t)row * F * 2, Z0, F * 2);
    }
}

static void experts_down(void)
{
    if (core) return;
    RouteMeta *meta = (RouteMeta *)META;
    for (unsigned e = 0; e < E; ++e)
        linear(HBM_EXPERT_OUTPUT + (uint64_t)meta->offset[e] * H * 2,
               HBM_ACTIVATION + (uint64_t)meta->offset[e] * F * 2,
               HBM_DOWN_WEIGHT[e], HBM_DOWN_BIAS[e], meta->count[e], F, H);
}

static void combine(void)
{
    if (core) return;
    RouteMeta *meta = (RouteMeta *)META;
    half *row_data = (half *)X0, *out = (half *)Z0, *probs = (half *)BIAS;
    float *acc = (float *)W0;
    copy(BIAS, HBM_ROUTE_WEIGHTS, batch * TOP * 2);
    for (unsigned b = cid; b < batch; b += ARCH_NUM_CLUSTER) {
        copy(X0, HBM_RESIDUAL + (uint64_t)b * H * 2, H * 2);
        for (unsigned j = 0; j < H; ++j) acc[j] = hfloat(row_data[j]);
        for (unsigned r = 0; r < batch * TOP; ++r) if (meta->token[r] == b) {
            float prob = hfloat(probs[b * TOP + meta->rank[r]]);
            copy(X0, HBM_EXPERT_OUTPUT + (uint64_t)r * H * 2, H * 2);
            for (unsigned j = 0; j < H; ++j) acc[j] += prob * hfloat(row_data[j]);
        }
        for (unsigned j = 0; j < H; ++j) out[j] = fhalf(acc[j]);
        copy(HBM_OUTPUT + (uint64_t)b * H * 2, Z0, H * 2);
    }
}

static void dump(uint64_t address, unsigned size)
{
    for (unsigned offset = 0; offset < size; offset += 65536) {
        unsigned n = size - offset; if (n > 65536) n = 65536;
        flex_dump_hbm(address - ARCH_HBM_START_BASE + offset, n);
    }
}

int main(void)
{
    cid = flex_get_cluster_id();
    flex_barrier_xy_init(); flex_global_barrier_xy();
    if (core == 0) copy(X0, HBM_CONTROL, 64);
    flex_intra_cluster_sync();
    unsigned *cfg = (unsigned *)X0;
    batch = cfg[1]; context = cfg[2]; dump_detail = cfg[3];
    if (cfg[0] != 0x47505431 || !batch || batch > MODEL_MAX_BATCH || context != MODEL_CACHED_TOKENS) {
        if (!cid && !core) { flex_print("DECODE_FAIL invalid control block\n"); flex_eoc(1); }
        return 1;
    }
    flex_global_barrier_xy();
    if (!cid && !core) { flex_timer_start(); previous = cycle(); }
    flex_global_barrier_xy();
    norm(HBM_NORM1_OUT, HBM_INPUT, HBM_NORM1); phase(0);
    linear(HBM_QKV, HBM_NORM1_OUT, HBM_QKV_WEIGHT, HBM_QKV_BIAS, batch, H, QKV_SIZE); phase(1);
    rotate_and_cache(); phase(2);
    attention(); phase(3);
    linear(HBM_PROJECTION, HBM_ATTENTION, HBM_OUT_WEIGHT, HBM_OUT_BIAS, batch, Q_SIZE, H); phase(4);
    residual(); phase(5);
    norm(HBM_NORM2_OUT, HBM_RESIDUAL, HBM_NORM2); phase(6);
    linear(HBM_ROUTER_LOGITS, HBM_NORM2_OUT, HBM_ROUTER_WEIGHT, HBM_ROUTER_BIAS, batch, H, E); phase(7);
    route(); phase(8);
    dispatch(); phase(9);
    experts_up(); phase(10);
    swiglu(); phase(11);
    experts_down(); phase(12);
    combine(); phase(13);
    if (!cid && !core) {
        uint64_t elapsed = 0;
        for (unsigned i = 0; i < 14; ++i) elapsed += ticks[i];
        flex_timer_end();
        static const char *names[] = {"norm1", "qkv", "rope_cache", "gqa", "out_projection",
            "attention_residual", "norm2", "router_gemm", "topk", "dispatch", "expert_up", "swiglu", "expert_down", "combine"};
        for (unsigned i = 0; i < 14; ++i) {
            flex_print("STAGE "); flex_print((char *)names[i]); flex_print(" "); number(ticks[i]); flex_print("\n");
        }
        unsigned distinct = 0;
        for (unsigned e = 0; e < E; ++e) if (((RouteMeta *)META)->count[e]) ++distinct;
        flex_print("ROUTING distinct_experts "); number(distinct); flex_print(" assignments "); number(batch * TOP); flex_print("\n");
        flex_print("LAYER_CYCLES "); number(elapsed); flex_print("\n");
        flex_dump_open();
        dump(HBM_OUTPUT, batch * H * 2);
        dump(HBM_ROUTES, batch * TOP * 4);
        dump(HBM_ROUTE_WEIGHTS, batch * TOP * 2);
        if (dump_detail) {
            dump(HBM_NORM1_OUT, batch * H * 2); dump(HBM_QKV, batch * QKV_SIZE * 2);
            dump(HBM_ATTENTION, batch * Q_SIZE * 2); dump(HBM_PROJECTION, batch * H * 2);
            dump(HBM_RESIDUAL, batch * H * 2); dump(HBM_NORM2_OUT, batch * H * 2);
            dump(HBM_ROUTER_LOGITS, batch * E * 2); dump(HBM_METADATA, sizeof(RouteMeta));
            dump(HBM_DISPATCH, batch * TOP * H * 2); dump(HBM_GATE_UP, batch * TOP * F * 4);
            dump(HBM_ACTIVATION, batch * TOP * F * 2); dump(HBM_EXPERT_OUTPUT, batch * TOP * H * 2);
        }
        flex_dump_close();
        flex_print("DECODE_DONE\n"); flex_eoc(0);
    }
    return 0;
}
