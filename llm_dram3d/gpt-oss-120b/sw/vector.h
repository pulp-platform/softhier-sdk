// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#ifndef LLM_VECTOR_H
#define LLM_VECTOR_H

/* Strip mining and vfexp follow implementation/sw/{RMSNorm,Activation,
 * FlatAttention}. FP32 intermediates avoid FP16 reduction overflow/loss.
 * Explicit clobbers and memory ordering keep these kernels safe under -O3.
 * The target has widening addition but no vfwcvt.f.f.v: add FP16 zero to
 * widen exactly, including subnormals. e16,m2 and e32,m4 have identical VLMAX.
 */
typedef _Float16 half_t;
#define VREGS "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", \
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory"

static void vector_sync(void)
{
    /* A scalar load interlocks with outstanding Spatz vector stores. The DMA
       offload itself does not provide this scalar/vector memory dependency. */
    __asm__ volatile("lw t0, 0(%0)\naddi t0, t0, 0\nfence rw, rw"
                     :: "r"(L_L1_A) : "t0", "memory");
}

static void v_widen(float *dst, const half_t *src, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vle16.v v0, (%2)\nvmv.v.i v4, 0\nvfwadd.vv v8, v0, v4\n"
            "vsetvli zero, %0, e32, m4, ta, ma\nvse32.v v8, (%3)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(dst) : VREGS);
        n -= vl; src += vl; dst += vl;
    }
    vector_sync();
}

static void v_narrow(half_t *dst, const float *src, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\n"
            "vsetvli zero, %0, e16, m2, ta, ma\nvfncvt.f.f.w v0, v8\nvse16.v v0, (%3)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(dst) : VREGS);
        n -= vl; src += vl; dst += vl;
    }
    vector_sync();
}

/* Step 5: keep the FP32 sum in registers until the single FP16 rounding. */
static void v_add_half(half_t *dst, const half_t *a, const half_t *b, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vle16.v v0, (%2)\nvle16.v v4, (%3)\nvfwadd.vv v8, v0, v4\n"
            "vfncvt.f.f.w v0, v8\nvse16.v v0, (%4)"
            : "=&r"(vl) : "r"(n), "r"(a), "r"(b), "r"(dst) : VREGS);
        n -= vl; a += vl; b += vl; dst += vl;
    }
    vector_sync();
}

/* dst = a * scale + b. Separate multiply/add also match the reference. */
static void v_axpby(float *dst, const float *a, float scale, const float *b, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\nvle32.v v12, (%3)\n"
            "vfmul.vf v8, v8, %5\nvfadd.vv v8, v8, v12\nvse32.v v8, (%4)"
            : "=&r"(vl) : "r"(n), "r"(a), "r"(b), "r"(dst), "f"(scale) : VREGS);
        n -= vl; a += vl; b += vl; dst += vl;
    }
    vector_sync();
}

/* Step 21: accumulate a weighted FP16 row without a widened memory temporary. */
static void v_acc_half(float *dst, const half_t *src, float weight, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vle16.v v0, (%2)\nvmv.v.i v4, 0\nvfwadd.vv v8, v0, v4\n"
            "vsetvli zero, %0, e32, m4, ta, ma\nvle32.v v12, (%3)\n"
            "vfmul.vf v8, v8, %4\nvfadd.vv v8, v8, v12\nvse32.v v8, (%3)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(dst), "f"(weight) : VREGS);
        n -= vl; src += vl; dst += vl;
    }
    vector_sync();
}

static void v_scale(float *dst, const float *src, float scale, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\n"
            "vfmul.vf v8, v8, %4\nvse32.v v8, (%3)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(dst), "f"(scale) : VREGS);
        n -= vl; src += vl; dst += vl;
    }
    vector_sync();
}

static float v_reduce(const float *src, unsigned n, float initial, unsigned maximum)
{
    float result = initial;
    while (n) {
        unsigned vl;
        if (maximum) {
            __asm__ volatile(
                "vsetvli %0, %2, e32, m4, ta, ma\nvfmv.s.f v0, %4\n"
                "vle32.v v8, (%3)\nvfredmax.vs v0, v8, v0\nvfmv.f.s %1, v0"
                : "=&r"(vl), "=f"(result) : "r"(n), "r"(src), "f"(result) : VREGS);
        } else {
            __asm__ volatile(
                "vsetvli %0, %2, e32, m4, ta, ma\nvfmv.s.f v0, %4\n"
                "vle32.v v8, (%3)\nvfredusum.vs v0, v8, v0\nvfmv.f.s %1, v0"
                : "=&r"(vl), "=f"(result) : "r"(n), "r"(src), "f"(result) : VREGS);
        }
        n -= vl; src += vl;
    }
    return result;
}

static void v_exp(float *dst, const float *src, float shift, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\n"
            "vfsub.vf v8, v8, %4\n.word 0x32041857\nvse32.v v16, (%3)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(dst), "f"(shift) : VREGS);
        n -= vl; src += vl; dst += vl;
    }
    vector_sync();
}

static void v_rmsnorm(half_t *dst, const half_t *src, const half_t *weight,
                      unsigned n, float *scratch)
{
    float *x = scratch, *w = x + n, *square = w + n;
    float sum = 0.0f, inv;
    if (!OPT(19)) { v_widen(x, src, n); v_widen(w, weight, n); }
    for (unsigned j = 0; j < n;) {
        unsigned vl;
        if (OPT(19)) {
            /* Preserve the original strip sizes and FP32 reduction order. */
            __asm__ volatile(
                "vsetvli %0, %2, e16, m2, ta, ma\n"
                "vle16.v v0, (%3)\nvmv.v.i v4, 0\nvfwadd.vv v8, v0, v4\n"
                "vsetvli zero, %0, e32, m4, ta, ma\nvfmul.vv v8, v8, v8\n"
                "vfmv.s.f v0, %4\nvfredusum.vs v0, v8, v0\nvfmv.f.s %1, v0"
                : "=&r"(vl), "=f"(sum) : "r"(n - j), "r"(src + j), "f"(sum) : VREGS);
        } else __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\n"
            "vfmul.vv v8, v8, v8\nvse32.v v8, (%3)"
            : "=&r"(vl) : "r"(n - j), "r"(x + j), "r"(square + j) : VREGS);
        j += vl;
    }
    vector_sync();
    if (!OPT(19)) sum = v_reduce(square, n, 0.0f, 0);
    __asm__ volatile(
        "vsetvli zero, %5, e32, m1, ta, ma\nvfmv.s.f v8, %1\n"
        "vfmul.vf v8, v8, %2\nvfadd.vf v8, v8, %3\nvfsqrt.v v8, v8\n"
        "vfmv.s.f v12, %4\nvfdiv.vv v8, v12, v8\nvfmv.f.s %0, v8"
        : "=f"(inv) : "f"(sum), "f"(1.0f / n), "f"(1e-5f), "f"(1.0f), "r"(1) : VREGS);
    for (unsigned j = 0; j < n;) {
        unsigned vl;
        if (OPT(19)) __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vle16.v v0, (%2)\nvle16.v v2, (%3)\nvmv.v.i v4, 0\n"
            "vfwadd.vv v8, v0, v4\nvfwadd.vv v12, v2, v4\n"
            "vsetvli zero, %0, e32, m4, ta, ma\n"
            "vfmul.vf v8, v8, %4\nvfmul.vv v8, v8, v12\n"
            "vsetvli zero, %0, e16, m2, ta, ma\nvfncvt.f.f.w v0, v8\nvse16.v v0, (%5)"
            : "=&r"(vl) : "r"(n - j), "r"(src + j), "r"(weight + j), "f"(inv), "r"(dst + j) : VREGS);
        else __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\nvle32.v v8, (%2)\nvle32.v v12, (%3)\n"
            "vfmul.vf v8, v8, %4\nvfmul.vv v8, v8, v12\n"
            "vsetvli zero, %0, e16, m2, ta, ma\nvfncvt.f.f.w v0, v8\nvse16.v v0, (%5)"
            : "=&r"(vl) : "r"(n - j), "r"(x + j), "r"(w + j), "f"(inv), "r"(dst + j) : VREGS);
        j += vl;
    }
    vector_sync();
}

static void v_rope(half_t *x, const half_t *table, unsigned dim, float *scratch)
{
    unsigned half = dim / 2;
    float *a = scratch, *b = a + half, *co = b + half, *si = co + half;
    if (!OPT(20)) {
        v_widen(a, x, half); v_widen(b, x + half, half);
        v_widen(co, table, half); v_widen(si, table + half, half);
    }
    for (unsigned j = 0; j < half;) {
        unsigned vl;
        if (OPT(20)) __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vle16.v v0, (%2)\nvle16.v v2, (%3)\nvle16.v v4, (%4)\nvle16.v v6, (%5)\n"
            "vmv.v.i v24, 0\nvfwadd.vv v8, v0, v24\nvfwadd.vv v12, v2, v24\n"
            "vfwadd.vv v16, v4, v24\nvfwadd.vv v20, v6, v24\n"
            "vsetvli zero, %0, e32, m4, ta, ma\n"
            "vfmul.vv v24, v8, v16\nvfmul.vv v28, v12, v20\nvfsub.vv v24, v24, v28\n"
            "vfmul.vv v8, v8, v20\nvfmul.vv v12, v12, v16\nvfadd.vv v8, v8, v12\n"
            "vsetvli zero, %0, e16, m2, ta, ma\n"
            "vfncvt.f.f.w v0, v24\nvse16.v v0, (%2)\nvfncvt.f.f.w v4, v8\nvse16.v v4, (%3)"
            : "=&r"(vl) : "r"(half - j), "r"(x + j), "r"(x + half + j),
                          "r"(table + j), "r"(table + half + j) : VREGS);
        else __asm__ volatile(
            "vsetvli %0, %1, e32, m4, ta, ma\n"
            "vle32.v v8, (%2)\nvle32.v v12, (%3)\nvle32.v v16, (%4)\nvle32.v v20, (%5)\n"
            "vfmul.vv v24, v8, v16\nvfmul.vv v28, v12, v20\nvfsub.vv v24, v24, v28\n"
            "vfmul.vv v8, v8, v20\nvfmul.vv v12, v12, v16\nvfadd.vv v8, v8, v12\n"
            "vsetvli zero, %0, e16, m2, ta, ma\n"
            "vfncvt.f.f.w v0, v24\nvse16.v v0, (%6)\nvfncvt.f.f.w v4, v8\nvse16.v v4, (%7)"
            : "=&r"(vl) : "r"(half - j), "r"(a + j), "r"(b + j), "r"(co + j), "r"(si + j),
                          "r"(x + j), "r"(x + half + j) : VREGS);
        j += vl;
    }
    vector_sync();
}

static void v_swiglu(half_t *dst, const half_t *src, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vlse16.v v0, (%2), %5\nvlse16.v v2, (%3), %5\nvmv.v.i v4, 0\n"
            "vfwadd.vv v8, v0, v4\nvfwadd.vv v24, v2, v4\n"
            "vsetvli zero, %0, e32, m4, ta, ma\n"
            "vfmin.vf v20, v8, %6\nvfmin.vf v24, v24, %6\nvfmax.vf v24, v24, %7\n"
            "vfmul.vf v8, v20, %8\n.word 0x32041857\nvfadd.vf v16, v16, %9\n"
            "vfdiv.vv v8, v20, v16\nvfadd.vf v24, v24, %9\nvfmul.vv v8, v8, v24\n"
            "vsetvli zero, %0, e16, m2, ta, ma\nvfncvt.f.f.w v0, v8\nvse16.v v0, (%4)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(src + 1), "r"(dst), "r"(4),
                          "f"(7.0f), "f"(-7.0f), "f"(-1.702f), "f"(1.0f) : VREGS);
        n -= vl; src += 2 * vl; dst += vl;
    }
    vector_sync();
}

/* Step 24: retain the FP16 bias-rounding boundary inside the fused kernel. */
static void v_bias_swiglu(half_t *dst, const half_t *src, const half_t *bias, unsigned n)
{
    while (n) {
        unsigned vl;
        __asm__ volatile(
            "vsetvli %0, %1, e16, m2, ta, ma\n"
            "vlse16.v v0, (%2), %7\nvlse16.v v2, (%3), %7\n"
            "vlse16.v v4, (%4), %7\nvlse16.v v6, (%5), %7\n"
            "vfwadd.vv v8, v0, v4\nvfwadd.vv v24, v2, v6\n"
            "vfncvt.f.f.w v0, v8\nvfncvt.f.f.w v2, v24\nvmv.v.i v4, 0\n"
            "vfwadd.vv v8, v0, v4\nvfwadd.vv v24, v2, v4\n"
            "vsetvli zero, %0, e32, m4, ta, ma\n"
            "vfmin.vf v20, v8, %8\nvfmin.vf v24, v24, %8\nvfmax.vf v24, v24, %9\n"
            "vfmul.vf v8, v20, %10\n.word 0x32041857\nvfadd.vf v16, v16, %11\n"
            "vfdiv.vv v8, v20, v16\nvfadd.vf v24, v24, %11\nvfmul.vv v8, v8, v24\n"
            "vsetvli zero, %0, e16, m2, ta, ma\nvfncvt.f.f.w v0, v8\nvse16.v v0, (%6)"
            : "=&r"(vl) : "r"(n), "r"(src), "r"(src + 1), "r"(bias), "r"(bias + 1),
                          "r"(dst), "r"(4), "f"(7.0f), "f"(-7.0f), "f"(-1.702f), "f"(1.0f) : VREGS);
        n -= vl; src += 2 * vl; bias += 2 * vl; dst += vl;
    }
    vector_sync();
}
#endif
