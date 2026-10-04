// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "vector.h"

int main(void)
{
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        half *in = (half *)local(0x1000), *out = (half *)local(0x3000);
        for (unsigned i = 0; i < 1027; ++i) { in[i] = fhalf(0.5f); out[i] = 0; }
        vec_scale(out, in, 1027, fhalf(3.0f));
        unsigned errors = 0;
        for (unsigned i = 0; i < 1027; ++i) if (out[i] != fhalf(1.5f)) {
            if (errors < 8) { flex_print("VEC mismatch "); flex_print_int(i); flex_print(" value "); flex_print_int(out[i]); flex_print("\n"); }
            ++errors;
        }
        for (unsigned i = 0; i < 1027; ++i) { in[i] = 0; out[i] = 0; }
        vec_exp(out, in, 1027);
        for (unsigned i = 0; i < 1027; ++i) if (out[i] != fhalf(1.0f)) ++errors;
        float *single = (float *)local(0x6000);
        for (unsigned n = 1; n <= 65; n += 4) {
            for (unsigned i = 0; i < n; ++i) single[i] = 0.0f;
            vec_exp32(single, n);
            for (unsigned i = 0; i < n; ++i) if (single[i] != 1.0f) ++errors;
        }
        // The KV-cache update gathers/scatters individual FP16 values. Verify both
        // halves of a bank word and preservation of neighboring sentinel values.
        for (unsigned i = 0; i < 32; ++i) { in[i] = i + 1; out[i] = 0x3555; }
        asm volatile("" ::: "memory");
        flex_dma_async_2d((uint32_t)out + 2, (uint32_t)in, 2, 6, 2, 8);
        flex_dma_async_wait_all();
        asm volatile("" ::: "memory");
        for (unsigned i = 0; i < 32; ++i) {
            half expected = i % 3 == 1 && i < 24 ? i / 3 + 1 : 0x3555;
            if (out[i] != expected) ++errors;
        }
        flex_print(errors ? "VECTOR_FAIL " : "VECTOR_PASS "); flex_print_int(errors); flex_print("\n");
    }
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_eoc(0);
    return 0;
}
