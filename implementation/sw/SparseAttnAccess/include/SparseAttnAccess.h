// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "flex_dma_pattern.h"

static inline uint32_t saa_cycles(void)
{
    uint32_t value;
    asm volatile("csrr %0, mcycle" : "=r"(value) :: "memory");
    return value;
}

static inline uint32_t saa_completed(void)
{
    register uint32_t value asm("a0");
    asm volatile(".word %1" : "=r"(value)
        : "i"(R_TYPE_ENCODE(DMSTATI_FUNCT7, 0, 0, XDMA_FUNCT3, 10, OP_CUSTOM1))
        : "memory");
    return value;
}

__attribute__((noinline, noclone))
static void saa_issue_1d(uint64_t dst, uint64_t src, uint32_t bytes)
{
    asm volatile("" ::: "memory");
    flex_dma_async_1d(dst, src, bytes);
    asm volatile("" ::: "memory");
}

__attribute__((always_inline)) static inline void
saa_issue_inline(uint32_t dst, uint64_t src, uint32_t bytes)
{
    register uint32_t d_lo asm("a0") = dst;
    register uint32_t d_hi asm("a1") = 0;
    register uint32_t s_lo asm("a2") = (uint32_t)src;
    register uint32_t s_hi asm("a3") = src >> 32;
    register uint32_t size asm("a4") = bytes;
    asm volatile(".word %5\n.word %6\n.word %7\n"
        : "+r"(d_lo)
        : "r"(d_hi), "r"(s_lo), "r"(s_hi), "r"(size),
          "i"(R_TYPE_ENCODE(DMSRC_FUNCT7, 13, 12, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMDST_FUNCT7, 11, 10, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMCPYI_FUNCT7, 0, 14, 0, 10, OP_CUSTOM1))
        : "memory");
}

// A physical token address divided by token_bytes is a 32-bit gather index.
// Source base zero preserves the logical selection order across HBM nodes.
__attribute__((always_inline)) static inline void
saa_issue_gather(uint32_t dst, const uint32_t *indices, uint32_t count, uint32_t token_bytes)
{
    register uint32_t d_lo asm("a0") = dst;
    register uint32_t d_hi asm("a1") = 0;
    register uint32_t s_lo asm("a2") = 0;
    register uint32_t s_hi asm("a3") = 0;
    register uint32_t size asm("a4") = token_bytes;
    register uint32_t index asm("a5") = (uint32_t)indices;
    register uint32_t src_stride asm("a6") = token_bytes;
    register uint32_t dst_stride asm("a7") = token_bytes;
    register uint32_t reps asm("t0") = count;
    asm volatile(".word %9\n.word %10\n.word %11\n.word %12\n.word %13\n.word %14\n"
        : "+r"(d_lo)
        : "r"(d_hi), "r"(s_lo), "r"(s_hi), "r"(size), "r"(index),
          "r"(src_stride), "r"(dst_stride), "r"(reps),
          "i"(R_TYPE_ENCODE(DMSRC_FUNCT7, 13, 12, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMDST_FUNCT7, 11, 10, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(8, 2, 15, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMSTR_FUNCT7, 17, 16, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMREP_FUNCT7, 0, 5, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMCPYI_FUNCT7, 4, 14, 0, 10, OP_CUSTOM1))
        : "memory");
}

static inline uint32_t saa_data_bits(uint32_t cluster, uint32_t token, uint32_t element)
{
    uint32_t value = token ^ (cluster * 0x9e3779b9U) ^ (element * 0x85ebca6bU) ^ 0xa5a55a5aU;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
#if SAA_ELEMENT_BYTES == 2
    return 0x3c00U | (value & 0x3ffU);
#else
    return 0x3f800000U | (value & 0x7fffffU);
#endif
}
