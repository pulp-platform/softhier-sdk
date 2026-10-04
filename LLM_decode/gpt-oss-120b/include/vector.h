// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "half.h"

static inline void vec_complete(const void *address)
{
    // Snitch's integer load interlock drains pending vector stores before scalar
    // FP loads or an independent DMA engine can consume their results.
    asm volatile("lw zero, 0(%0)" :: "r"((uint32_t)address & ~3u) : "memory");
}

// Use explicit register groups and compiler memory barriers around vector memory access.
// The custom exponential encoding is the same as implementation/FlatAttention.
static inline void vec_scale(half *out, const half *in, unsigned n, half scale)
{
    half *start = out;
    while (n) {
        unsigned vl;
        asm volatile(
            "vsetvli %0, %1, e16, m8, ta, ma\n"
            "fmv.w.x ft0, %4\n"
            "vle16.v v8, (%2)\n"
            "vfmul.vf v8, v8, ft0\n"
            "vse16.v v8, (%3)\n"
            : "=&r"(vl) : "r"(n), "r"(in), "r"(out), "r"(0xffff0000u | scale)
            : "ft0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
        n -= vl; in += vl; out += vl;
    }
    vec_complete(start);
}

static inline void vec_exp(half *out, const half *in, unsigned n)
{
    half *start = out;
    while (n) {
        unsigned vl;
        asm volatile(
            "vsetvli %0, %1, e16, m8, ta, ma\n"
            "vle16.v v8, (%2)\n"
            ".word 0x32041857\n"
            "vse16.v v16, (%3)\n"
            : "=&r"(vl) : "r"(n), "r"(in), "r"(out)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
        n -= vl; in += vl; out += vl;
    }
    vec_complete(start);
}

static inline void vec_add(half *out, const half *a, const half *b, unsigned n)
{
    half *start = out;
    while (n) {
        unsigned vl;
        asm volatile(
            "vsetvli %0, %1, e16, m8, ta, ma\n"
            "vle16.v v8, (%2)\n" "vle16.v v16, (%3)\n"
            "vfadd.vv v8, v8, v16\n" "vse16.v v8, (%4)\n"
            : "=&r"(vl) : "r"(n), "r"(a), "r"(b), "r"(out)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
        n -= vl; a += vl; b += vl; out += vl;
    }
    vec_complete(start);
}

static inline void vec_exp32(float *data, unsigned n)
{
    float *start = data;
    while (n) {
        unsigned vl;
        asm volatile(
            "vsetvli %0, %1, e32, m8, ta, ma\n"
            "vle32.v v8, (%2)\n" ".word 0x32041857\n" "vse32.v v16, (%2)\n"
            : "=&r"(vl) : "r"(n), "r"(data)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
        n -= vl; data += vl;
    }
    vec_complete(start);
}
