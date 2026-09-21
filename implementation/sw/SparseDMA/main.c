// SPDX-License-Identifier: Apache-2.0
#include "flex_runtime.h"
#include "sparse_dma.h"
#include "preload.h"
#include "golden.h"
#include "SparseDMA.h"

static uint32_t l1_indices[SPARSE_SELECTED] __attribute__((section(".l1"), aligned(8)));
static uint16_t l1_indices16[(SPARSE_SELECTED + 3) & ~3U]
    __attribute__((section(".l1"), aligned(8)));
static struct {
    uint16_t before[32];
    uint16_t rows[SPARSE_SELECTED * SPARSE_DIM];
    uint16_t after[32];
} output __attribute__((section(".l1"), aligned(64)));

int main(void)
{
    // All 16 clusters and 80 harts are instantiated. Only cluster 0/core 0
    // submits work; other harts park without generating benchmark traffic.
    if (flex_get_cluster_id() != 0 || flex_get_core_id() != 0) {
        while (1) asm volatile("wfi" ::: "memory");
    }

    uint32_t errors = 0;
    const uint32_t row_bytes = SPARSE_DIM * sizeof(uint16_t);
    bare_dma_set_mask(0, 0);
    sparse_issue_1d((uint32_t)l1_indices, SPARSE_INDEX_ADDR, sizeof(l1_indices));
    sparse_wait();
    for (uint32_t i = 0; i < SPARSE_SELECTED; ++i) {
        if (l1_indices[i] != sparse_expected_indices[i]) ++errors;
        l1_indices16[i] = (uint16_t)l1_indices[i];
    }
    for (uint32_t i = 0; i < 32; ++i) output.before[i] = output.after[i] = 0xa55a;
    for (uint32_t i = 0; i < SPARSE_SELECTED * SPARSE_DIM; ++i) output.rows[i] = 0xa55a;
    asm volatile("" ::: "memory");
    if (errors) {
        printf("SPARSE_DMA_FAIL variant=%s phase=index-preload errors=%u\n", SPARSE_VARIANT_NAME, errors);
        flex_eoc(1);
        return 1;
    }
    printf("SPARSE_DMA_START variant=%s cluster=0 core=0 hbm=west node=0 matrix=0x%08x index=0x%08x dst=0x%08x\n",
        SPARSE_VARIANT_NAME, (uint32_t)SPARSE_MATRIX_ADDR, (uint32_t)SPARSE_INDEX_ADDR, (uint32_t)output.rows);

    uint32_t completed_before = sparse_completed();
    uint32_t start = sparse_cycles();
#if SPARSE_VARIANT == 0
    for (uint32_t i = 0; i < SPARSE_SELECTED; ++i) {
        uint64_t src = SPARSE_MATRIX_ADDR + (uint64_t)l1_indices[i] * row_bytes;
        sparse_issue_1d((uint32_t)&output.rows[i * SPARSE_DIM], src, row_bytes);
    }
#elif SPARSE_VARIANT == 1
    uint32_t dst = (uint32_t)output.rows;
    for (uint32_t i = 0; i < SPARSE_SELECTED; ++i) {
        uint32_t src = (uint32_t)SPARSE_MATRIX_ADDR + l1_indices[i] * row_bytes;
        sparse_issue_inline(dst, src, row_bytes);
        dst += row_bytes;
    }
#else
    sparse_issue_gather((uint32_t)output.rows, SPARSE_MATRIX_ADDR, l1_indices16,
                       SPARSE_SELECTED, row_bytes);
#endif
    sparse_wait();
    uint32_t cycles = sparse_cycles() - start;
    uint32_t transfers = sparse_completed() - completed_before;

    // Host-generated reference contains every original FP16 bit. Verification
    // reads no HBM and is outside the measured issue-through-completion region.
    uint32_t checksum = 2166136261U;
    for (uint32_t i = 0; i < SPARSE_SELECTED * SPARSE_DIM; ++i) {
        uint16_t value = output.rows[i];
        if (value != sparse_golden[i]) {
            if (errors < 8) printf("SPARSE_DMA_MISMATCH item=%u got=0x%04x expected=0x%04x\n", i, value, sparse_golden[i]);
            ++errors;
        }
        checksum = (checksum ^ (value & 0xff)) * 16777619U;
        checksum = (checksum ^ (value >> 8)) * 16777619U;
    }
    for (uint32_t i = 0; i < 32; ++i)
        if (output.before[i] != 0xa55a || output.after[i] != 0xa55a) ++errors;
    if (transfers != (SPARSE_VARIANT == 2 ? 1 : SPARSE_SELECTED)) ++errors;
    if (checksum != SPARSE_EXPECTED_CHECKSUM) ++errors;
    printf("SPARSE_DMA_RESULT variant=%s rows=%u dim=%u selected=%u row_bytes=%u bytes=%u cycles=%u transfers=%u checksum=%08x errors=%u\n",
        SPARSE_VARIANT_NAME, SPARSE_ROWS, SPARSE_DIM, SPARSE_SELECTED, row_bytes,
        SPARSE_SELECTED * row_bytes, cycles, transfers, checksum, errors);
    printf("SPARSE_DMA_%s variant=%s\n", errors ? "FAIL" : "PASS", SPARSE_VARIANT_NAME);
    flex_eoc(errors != 0);
    return errors != 0;
}
