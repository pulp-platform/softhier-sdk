// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#include "gemm.h"
#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_redmule.h"
#include "flex_printf.h"

/* All DRAM addresses stay 64-bit up to the DMA custom instruction. */
static uint64_t channel(unsigned cid, unsigned offset)
{
    return (uint64_t)ARCH_DRAM3D_ADDR_BASE +
           (uint64_t)cid * ARCH_DRAM3D_NODE_SPACE + offset;
}

static void memory_fence(void)
{
    __asm__ volatile ("fence rw, rw" ::: "memory");
}

static unsigned cycles(void)
{
    unsigned value;
    __asm__ volatile ("rdcycle %0" : "=r"(value));
    return value;
}

static void panels(unsigned cid, unsigned mi, unsigned ni, unsigned step, unsigned clear_z)
{
    unsigned slot = step % D3_BUFFERS;
    unsigned xbuf = D3_L1_X + slot * D3_X_BYTES;
    unsigned wbuf = D3_L1_W + slot * D3_W_BYTES;
#if D3_SUMMA
    unsigned x = cid % ARCH_NUM_CLUSTER_X, y = cid / ARCH_NUM_CLUSTER_X;
    unsigned load_x = x == step % ARCH_NUM_CLUSTER_X;
    unsigned load_w = y == step % ARCH_NUM_CLUSTER_Y;
    unsigned ai = mi * (D3_STEPS / ARCH_NUM_CLUSTER_X) + step / ARCH_NUM_CLUSTER_X;
    unsigned bi = ni * (D3_STEPS / ARCH_NUM_CLUSTER_Y) + step / ARCH_NUM_CLUSTER_Y;
#else
    unsigned load_x = 1, load_w = 1;
    unsigned ai = mi * D3_STEPS + step;
    unsigned bi = ni * D3_STEPS + step;
#endif
    if (clear_z)
        flex_dma_async_1d(clear_z, zomem(0), D3_Z_BYTES);
    if (load_x)
        flex_dma_async_1d(xbuf, channel(cid, ai * D3_X_BYTES), D3_X_BYTES);
    if (load_w)
        flex_dma_async_1d(wbuf, channel(cid, D3_W_OFFSET + bi * D3_W_BYTES), D3_W_BYTES);
    flex_dma_async_wait_all();
#if D3_SUMMA
    /* A travels along a row, B along a column. A zero mask bit varies the
       corresponding coordinate; set bits keep it equal to the source. */
    if (load_x) {
        flex_dma_async_broadcast(xbuf - ARCH_CLUSTER_TCDM_BASE,
            xbuf - ARCH_CLUSTER_TCDM_BASE, D3_X_BYTES, 0, ARCH_NUM_CLUSTER_Y - 1);
        /* Drain before changing masks: the legacy DMA backend uses the active
           transfer's collective mask while emitting its outstanding bursts. */
        flex_dma_async_wait_all();
    }
    if (load_w) {
        flex_dma_async_broadcast(wbuf - ARCH_CLUSTER_TCDM_BASE,
            wbuf - ARCH_CLUSTER_TCDM_BASE, D3_W_BYTES, ARCH_NUM_CLUSTER_X - 1, 0);
        flex_dma_async_wait_all();
    }
#endif
    memory_fence();
}

static void gemm(unsigned cid)
{
#if D3_Z_BUFFERS > 1
    /* Keep the panel pipeline running across C-tile boundaries. Alternating
       C buffers allow the previous result to drain during the next GEMM. */
    if (flex_is_first_core())
        flex_redmule_config(D3_MT, D3_KT, D3_NT);
    if (flex_is_dm_core()) {
        panels(cid, 0, 0, 0, D3_L1_Z);
    }
    flex_global_barrier_xy();
    for (unsigned tile = 0; tile < D3_MB * D3_NB; ++tile) {
        unsigned zbuf = D3_L1_Z + (tile % D3_Z_BUFFERS) * D3_Z_BYTES;
        for (unsigned step = 0; step < D3_STEPS; ++step) {
            if (flex_is_first_core()) {
                memory_fence();
                flex_redmule_trigger(D3_L1_X + (step % D3_BUFFERS) * D3_X_BYTES,
                    D3_L1_W + (step % D3_BUFFERS) * D3_W_BYTES,
                    zbuf, REDMULE_FP_16);
                flex_redmule_wait();
                memory_fence();
            }
            if (flex_is_dm_core()) {
                unsigned store_step = 0;
#if D3_SUMMA
                /* The (1,1) root reads both panels for K step 1. Drain its
                   previous C one step later, when it reads neither panel. */
                if (cid % ARCH_NUM_CLUSTER_X == 1 && cid / ARCH_NUM_CLUSTER_X == 1)
                    store_step = 1;
#endif
                if (tile && step == store_step) {
                    flex_dma_async_1d(channel(cid, D3_Z_OFFSET + (tile - 1) * D3_Z_BYTES),
                        D3_L1_Z + ((tile - 1) % D3_Z_BUFFERS) * D3_Z_BYTES, D3_Z_BYTES);
                    /* Drain before switching the legacy DMA's read backend
                       from TCDM to AXI. RedMule continues on the other C buffer. */
                    flex_dma_async_wait_all();
                }
                if (step + 1 < D3_STEPS) {
                    panels(cid, tile / D3_NB, tile % D3_NB, step + 1, 0);
                } else if (tile + 1 < D3_MB * D3_NB) {
                    panels(cid, (tile + 1) / D3_NB, (tile + 1) % D3_NB, 0,
                        D3_L1_Z + ((tile + 1) % D3_Z_BUFFERS) * D3_Z_BYTES);
                }
            }
            flex_global_barrier_xy();
        }
    }
    if (flex_is_dm_core()) {
        unsigned last = D3_MB * D3_NB - 1;
        flex_dma_async_1d(channel(cid, D3_Z_OFFSET + last * D3_Z_BYTES),
            D3_L1_Z + (last % D3_Z_BUFFERS) * D3_Z_BYTES, D3_Z_BYTES);
        flex_dma_async_wait_all();
    }
    flex_global_barrier_xy();
#else
    for (unsigned mi = 0; mi < D3_MB; ++mi) {
        for (unsigned ni = 0; ni < D3_NB; ++ni) {
            if (flex_is_dm_core()) {
                panels(cid, mi, ni, 0, D3_L1_Z);
            }
            flex_global_barrier_xy();
            for (unsigned step = 0; step < D3_STEPS; ++step) {
                if (flex_is_first_core()) {
                    memory_fence();
                    /* Runtime argument order is M, K (reduction), N. */
                    flex_redmule_config(D3_MT, D3_KT, D3_NT);
                    flex_redmule_trigger(D3_L1_X + (step % D3_BUFFERS) * D3_X_BYTES,
                        D3_L1_W + (step % D3_BUFFERS) * D3_W_BYTES,
                        D3_L1_Z, REDMULE_FP_16);
                    flex_redmule_wait();
                    memory_fence();
                }
                /* The DM core fetches and broadcasts the next panels while
                   core 0 runs RedMule on the other buffer. */
                if (flex_is_dm_core() && step + 1 < D3_STEPS)
                    panels(cid, mi, ni, step + 1, 0);
                flex_global_barrier_xy();
            }
            if (flex_is_dm_core()) {
                flex_dma_async_1d(channel(cid, D3_Z_OFFSET +
                    (mi * D3_NB + ni) * D3_Z_BYTES), D3_L1_Z, D3_Z_BYTES);
                flex_dma_async_wait_all();
            }
            flex_global_barrier_xy();
        }
    }
#endif
}

static unsigned check(unsigned cid)
{
    unsigned errors = 0;
    for (unsigned tile = 0; tile < D3_MB * D3_NB; ++tile) {
        if (flex_is_dm_core()) {
            flex_dma_async_1d(D3_L1_X, channel(cid, D3_Z_OFFSET + tile * D3_Z_BYTES), D3_Z_BYTES);
            flex_dma_async_1d(D3_L1_W, channel(cid, D3_GOLD_OFFSET + tile * D3_Z_BYTES), D3_Z_BYTES);
            flex_dma_async_wait_all();
            memory_fence();
        }
        flex_intra_cluster_sync();
        if (flex_is_first_core()) {
            volatile uint16_t *actual = (volatile uint16_t *)D3_L1_X;
            volatile uint16_t *gold = (volatile uint16_t *)D3_L1_W;
            for (unsigned i = 0; i < D3_Z_BYTES / 2; ++i) {
                unsigned a = actual[i], b = gold[i];
                if (a != b && ((a | b) & 0x7fff)) {
                    if (errors == 0) {
                        *(volatile unsigned *)(ARCH_SYNC_BASE + cid * ARCH_SYNC_SIZE + 36) =
                            tile * (D3_Z_BYTES / 2) + i;
                        *(volatile unsigned *)(ARCH_SYNC_BASE + cid * ARCH_SYNC_SIZE + 40) =
                            (a << 16) | b;
                    }
                    ++errors;
                }
            }
        }
        flex_intra_cluster_sync();
    }
    return errors;
}

int main(void)
{
    unsigned cid = flex_get_cluster_id();
    unsigned begin = 0, elapsed = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (cid == 0 && flex_is_first_core()) {
        flex_timer_start();
        begin = cycles();
    }
    flex_global_barrier_xy();
    gemm(cid);
    if (cid == 0 && flex_is_first_core()) {
        elapsed = cycles() - begin;
        flex_timer_end();
    }
    flex_global_barrier_xy();

    /* Validation reads back DRAM, outside the measurement interval. */
    unsigned errors = check(cid);
    if (flex_is_first_core())
        *(volatile unsigned *)(ARCH_SYNC_BASE + cid * ARCH_SYNC_SIZE + 32) = errors;
    flex_global_barrier_xy();
    if (cid == 0 && flex_is_first_core()) {
        errors = 0;
        for (unsigned i = 0; i < ARCH_NUM_CLUSTER; ++i) {
            unsigned count = *(volatile unsigned *)(ARCH_SYNC_BASE + i * ARCH_SYNC_SIZE + 32);
            errors += count;
            if (count) {
                unsigned elem = *(volatile unsigned *)(ARCH_SYNC_BASE + i * ARCH_SYNC_SIZE + 36);
                unsigned values = *(volatile unsigned *)(ARCH_SYNC_BASE + i * ARCH_SYNC_SIZE + 40);
                printf("DRAM3D_MISMATCH cid=%u elem=%u actual=%x expected=%x errors=%u\n",
                       i, elem, values >> 16, values & 0xffff, count);
            }
        }
        printf("DRAM3D_GEMM_%s M=%u N=%u K=%u cycles=%u errors=%u checked=%u\n",
               errors ? "FAIL" : "PASS", D3_M, D3_N, D3_K, elapsed, errors, D3_M * D3_N);
        flex_eoc(errors != 0);
    }
    return 0;
}
