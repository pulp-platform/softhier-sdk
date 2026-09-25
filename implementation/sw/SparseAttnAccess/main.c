// SPDX-License-Identifier: Apache-2.0
#include "flex_runtime.h"
#include "sparse_attn_access.h"
#include "SparseAttnAccess.h"

struct {
    uint32_t meta[8];
    uint32_t tokens[SAA_PAD_SELECTED];
    uint32_t address_indices[SAA_PAD_SELECTED];
} saa_input __attribute__((section(".l1"), aligned(64)));

static struct {
    uint32_t before[16];
    SAA_STORAGE_TYPE data[SAA_SELECTED * SAA_TOKEN_DIM];
    uint32_t after[16];
} saa_output __attribute__((section(".l1"), aligned(64)));

static uint32_t fnv_word(uint32_t hash, uint32_t value, uint32_t bytes)
{
    for (uint32_t i = 0; i < bytes; ++i) {
        hash = (hash ^ (value & 0xffU)) * 16777619U;
        value >>= 8;
    }
    return hash;
}

int main(void)
{
    const uint32_t cid = flex_get_cluster_id();
    const uint32_t core = flex_get_core_id();
    uint32_t errors = 0, start = 0, finish = 0, transfers = 0, checksum = 2166136261U;
    flex_barrier_init();
    if (core == 0) {
        if (saa_input.meta[0] != 0x53414131U || saa_input.meta[1] != cid ||
            saa_input.meta[2] != SAA_SELECTED || saa_input.meta[3] != SAA_CONTEXT ||
            saa_input.meta[4] != SAA_TOKEN_BYTES || saa_input.meta[5] != SAA_SEED) ++errors;
        uint32_t index_hash = 2166136261U;
        for (uint32_t i = 0; i < SAA_SELECTED; ++i) {
            const uint32_t token = saa_input.tokens[i], node = token % SAA_HBM_NODES;
            const uint32_t local_tokens = (SAA_CONTEXT + SAA_HBM_NODES - 1 - node) / SAA_HBM_NODES;
            uint64_t address = SAA_HBM_BASE + (uint64_t)node * SAA_HBM_WINDOW + SAA_HBM_OFFSET +
                ((uint64_t)cid * local_tokens + token / SAA_HBM_NODES) * SAA_TOKEN_BYTES;
            if (token >= SAA_CONTEXT || address != ((uint64_t)saa_input.address_indices[i] << SAA_TOKEN_SHIFT)) ++errors;
            if (SAA_SORTED && i && saa_input.tokens[i-1] >= token) ++errors;
            index_hash = fnv_word(index_hash, token, 4);
            index_hash = fnv_word(index_hash, saa_input.address_indices[i], 4);
        }
        if (index_hash != saa_input.meta[7]) ++errors;
        if (errors) {
            printf("SAA_FAIL variant=%s cluster=%u phase=indices errors=%u\n", SAA_VARIANT_NAME, cid, errors);
            flex_eoc(1);
            return 1;
        }
        for (uint32_t i = 0; i < 16; ++i) saa_output.before[i] = saa_output.after[i] = 0xa55aa55aU;
        for (uint32_t i = 0; i < SAA_SELECTED * SAA_TOKEN_DIM; ++i) saa_output.data[i] = 0;
        bare_dma_set_mask(0, 0);
        if (cid == 0) printf("SAA_RUN_BEGIN variant=%s clusters=%u N=%u M=%u token_bytes=%u batch=%u\n",
            SAA_VARIANT_NAME, SAA_CLUSTERS, SAA_SELECTED, SAA_CONTEXT, SAA_TOKEN_BYTES, SAA_BATCH);
    }
    flex_global_barrier();
    if (core == 0) {
        uint32_t completed_before = saa_completed();
        // The existing cluster annotation register supplies global-time ROI markers.
        flex_annotate_barrier(SAA_ROI_TYPE);
        start = saa_cycles();
        for (uint32_t first = 0; first < SAA_SELECTED; first += SAA_BATCH) {
            uint32_t count = SAA_SELECTED - first;
            if (count > SAA_BATCH) count = SAA_BATCH;
#if SAA_VARIANT == 2
            saa_issue_gather((uint32_t)&saa_output.data[first * SAA_TOKEN_DIM],
                &saa_input.address_indices[first], count, SAA_TOKEN_BYTES);
#else
            for (uint32_t i = first; i < first + count; ++i) {
                uint64_t src = (uint64_t)saa_input.address_indices[i] << SAA_TOKEN_SHIFT;
                uint32_t dst = (uint32_t)&saa_output.data[i * SAA_TOKEN_DIM];
#if SAA_VARIANT == 0
                saa_issue_1d(dst, src, SAA_TOKEN_BYTES);
#else
                saa_issue_inline(dst, src, SAA_TOKEN_BYTES);
#endif
            }
#endif
            bare_dma_wait_all();
            asm volatile("" ::: "memory");
        }
        finish = saa_cycles();
        flex_annotate_barrier(SAA_ROI_TYPE);
        transfers = saa_completed() - completed_before;
    }
    // Keep output verification outside every cluster's measured execution window.
    flex_global_barrier();
    if (core == 0) {
        for (uint32_t row = 0; row < SAA_SELECTED; ++row) {
            for (uint32_t element = 0; element < SAA_TOKEN_DIM; ++element) {
                uint32_t value = saa_output.data[row * SAA_TOKEN_DIM + element];
                if (value != saa_data_bits(cid, saa_input.tokens[row], element)) ++errors;
                checksum = fnv_word(checksum, value, SAA_ELEMENT_BYTES);
            }
        }
        for (uint32_t i = 0; i < 16; ++i)
            if (saa_output.before[i] != 0xa55aa55aU || saa_output.after[i] != 0xa55aa55aU) ++errors;
        if (transfers != (SAA_VARIANT == 2 ? (SAA_SELECTED + SAA_BATCH - 1) / SAA_BATCH : SAA_SELECTED)) ++errors;
        if (checksum != saa_input.meta[6]) ++errors;
    }
    // The SDK console is shared by all clusters. Serialize result lines after
    // verification so character writes cannot interleave across users.
    for (uint32_t owner = 0; owner < SAA_CLUSTERS; ++owner) {
        flex_global_barrier();
        if (core == 0 && cid == owner)
            printf("SAA_RESULT variant=%s cluster=%u start=%u end=%u cycles=%u transfers=%u checksum=%08x errors=%u\n",
                SAA_VARIANT_NAME, cid, start, finish, finish-start, transfers, checksum, errors);
    }
    flex_global_barrier();
    if (cid == 0 && core == 0) {
        printf("SAA_RUN_DONE variant=%s clusters=%u\n", SAA_VARIANT_NAME, SAA_CLUSTERS);
        flex_eoc(0);
    }
    return 0;
}
