// Copyright (C) 2026 ETH Zurich and University of Bologna
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

typedef uint16_t half;

static inline float hfloat(half h)
{
    union { uint32_t u; float f; } v;
    unsigned e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) return (h & 0x8000 ? -1.0f : 1.0f) * (float)m * 0x1p-24f;
    v.u = ((uint32_t)(h & 0x8000) << 16) | ((e == 31 ? 255 : e + 112) << 23) | (m << 13);
    return v.f;
}

static inline half fhalf(float f)
{
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000, mant = v.u & 0x7fffff;
    int exp = (int)((v.u >> 23) & 255) - 112;
    if (exp >= 31) return sign | (exp == 143 && mant ? 0x7e00 : 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000;
        unsigned shift = 14 - exp;
        unsigned rounded = (mant + ((1u << (shift - 1)) - 1) + ((mant >> shift) & 1)) >> shift;
        return sign | rounded;
    }
    mant += 0xfff + ((mant >> 13) & 1);
    return sign | (((uint32_t)exp << 10) + (mant >> 13));
}
