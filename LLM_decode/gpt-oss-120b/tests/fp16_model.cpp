// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#include "fp16.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

int main()
{
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00 && (h & 0x3ff)) continue;
        assert(soft_hier_float_to_fp16(soft_hier_fp16_to_float(h)) == h);
    }
    assert(soft_hier_fp16_to_float(1) == 0x1p-24f);
    assert(soft_hier_float_to_fp16(0x1p-24f) == 1);
    assert(soft_hier_float_to_fp16(0x1p-25f) == 0);
    assert(soft_hier_float_to_fp16(-0x1p-24f) == 0x8001);
    assert(soft_hier_float_to_fp16(1.0f + 0x1p-11f) == 0x3c00);
    assert(soft_hier_float_to_fp16(1.0f + 3 * 0x1p-11f) == 0x3c02);
    assert(soft_hier_float_to_fp16(65504.0f) == 0x7bff);
    assert(soft_hier_float_to_fp16(65520.0f) == 0x7c00);
    assert(std::isnan(soft_hier_fp16_to_float(0x7e00)));
    puts("FP16_MODEL_PASS");
}
