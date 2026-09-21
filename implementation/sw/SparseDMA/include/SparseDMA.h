// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "flex_dma_pattern.h"

static inline uint32_t sparse_cycles(void)
{
    uint32_t value;
    asm volatile("csrr %0, mcycle" : "=r"(value) :: "memory");
    return value;
}

static inline uint32_t sparse_completed(void)
{
    register uint32_t value asm("a0");
    asm volatile(".word %1" : "=r"(value)
        : "i"(R_TYPE_ENCODE(DMSTATI_FUNCT7, 0, 0, XDMA_FUNCT3, 10, OP_CUSTOM1))
        : "memory");
    return value;
}

static inline void sparse_wait(void)
{
    bare_dma_wait_all();
    asm volatile("" ::: "memory");
}

// Retain the out-of-line SDK call in the core-loop reference. Prevent GCC
// from turning it into the inlined variant or specializing its arguments.
__attribute__((noinline, noclone))
static void sparse_issue_1d(uint64_t dst, uint64_t src, uint32_t bytes)
{
    asm volatile("" ::: "memory");
    flex_dma_async_1d(dst, src, bytes);
    asm volatile("" ::: "memory");
}

__attribute__((always_inline)) static inline void
sparse_issue_inline(uint32_t dst, uint32_t src, uint32_t bytes)
{
    register uint32_t d_lo asm("a0") = dst;
    register uint32_t d_hi asm("a1") = 0;
    register uint32_t s_lo asm("a2") = src;
    register uint32_t s_hi asm("a3") = 0;
    register uint32_t size asm("a4") = bytes;
    asm volatile(
        ".word %5\n.word %6\n.word %7\n"
        : "+r"(d_lo)
        : "r"(d_hi), "r"(s_lo), "r"(s_hi), "r"(size),
          "i"(R_TYPE_ENCODE(DMSRC_FUNCT7, 13, 12, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMDST_FUNCT7, 11, 10, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMCPYI_FUNCT7, 0, 14, 0, 10, OP_CUSTOM1))
        : "memory");
}

__attribute__((always_inline)) static inline void
sparse_issue_gather(uint32_t dst, uint64_t src, const uint16_t *indices,
                    uint32_t count, uint32_t row_bytes)
{
    register uint32_t d_lo asm("a0") = dst;
    register uint32_t d_hi asm("a1") = 0;
    register uint32_t s_lo asm("a2") = (uint32_t)src;
    register uint32_t s_hi asm("a3") = src >> 32;
    register uint32_t size asm("a4") = row_bytes;
    register uint32_t index asm("a5") = (uint32_t)indices;
    register uint32_t src_stride asm("a6") = row_bytes;
    register uint32_t dst_stride asm("a7") = row_bytes;
    register uint32_t reps asm("t0") = count;
    asm volatile(
        ".word %9\n.word %10\n.word %11\n.word %12\n.word %13\n.word %14\n"
        : "+r"(d_lo)
        : "r"(d_hi), "r"(s_lo), "r"(s_hi), "r"(size), "r"(index),
          "r"(src_stride), "r"(dst_stride), "r"(reps),
          "i"(R_TYPE_ENCODE(DMSRC_FUNCT7, 13, 12, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMDST_FUNCT7, 11, 10, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(8, 1, 15, 0, 0, OP_CUSTOM1)), // DMIDX: 16-bit indices
          "i"(R_TYPE_ENCODE(DMSTR_FUNCT7, 17, 16, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMREP_FUNCT7, 0, 5, 0, 0, OP_CUSTOM1)),
          "i"(R_TYPE_ENCODE(DMCPYI_FUNCT7, 4, 14, 0, 10, OP_CUSTOM1))
        : "memory");
}
