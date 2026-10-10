#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Pack FP16 GEMM panels into independent stacked channels and emit an ELF64."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

CASES = {
    "big1024": (1024, 1024, 1024),
    "big2048": (1024, 2048, 1024),
    "flat16": (16, 1024, 1024),
    "flat8": (8, 2048, 1024),
}


def load_arch(path):
    spec = importlib.util.spec_from_file_location("dram3d_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlexClusterArch()


def elf64(path, regions):
    """Physical addresses are 64-bit even though the executing cores are RV32."""
    headers, payload = bytearray(), bytearray()
    offset = 64 + 56 * len(regions)
    for address, data in regions:
        headers += struct.pack("<IIQQQQQQ", 1, 6, offset, address, address,
                               len(data), len(data), 1)
        payload += data
        offset += len(data)
    path.write_bytes(b"\x7fELF\x02\x01\x01" + bytes(9) +
                     struct.pack("<HHIQQQIHHHHHH", 2, 243, 1, 0, 64, 0, 0,
                                 64, 56, len(regions), 0, 0, 0) + headers + payload)


def align(value):
    return (value + 127) & ~127


def generate(args):
    arch = load_arch(args.arch)
    m, n, k = CASES[args.case]
    nx, ny = arch.num_cluster_x, arch.num_cluster_y
    clusters = nx * ny
    flat = m < 32
    if flat and args.dataflow == "summa":
        raise ValueError("The flat presets use N partitioning; select auto or local")
    summa = not flat and args.dataflow != "local"
    if (nx, ny) != (4, 4) or arch.num_core_per_cluster < 2:
        raise ValueError("These presets require the default 4x4 grid and at least two cores")
    if not arch.dram3d_enable or any(arch.hbm_chan_placement):
        raise ValueError("Select a stacked-only architecture")
    if flat:
        mt, nt, kt = m, n // clusters, k
    else:
        mt, nt, kt = (128, 128, 256) if summa else (256, 128, 128)
    mb, nb, steps = (1, 1, 1) if flat else (m // (ny * mt), n // (nx * nt), k // kt)
    buffers = 1 if flat else 2
    zbuffers = 1 if flat else 2
    xb, wb, zb = 2 * mt * kt, 2 * kt * nt, 2 * mt * nt
    acount = mb * (steps // nx if summa else steps)
    bcount = nb * (steps // ny if summa else steps)
    woff = align(acount * xb)
    zoff = align(woff + bcount * wb)
    goldoff = align(zoff + mb * nb * zb)
    end = goldoff + mb * nb * zb
    if end > arch.dram3d_node_space:
        raise ValueError("Packed matrices do not fit in a channel aperture")
    lx = arch.cluster_tcdm_base + 0x1000
    lw = lx + buffers * xb
    lz = lw + buffers * wb
    if lz + zbuffers * zb + 512 > arch.cluster_tcdm_base + arch.cluster_tcdm_size:
        raise ValueError("GEMM buffers exceed TCDM")

    # Products are multiples of 1/64 and every partial sum fits exactly in FP16
    # for K=1024. Exact checking therefore detects even one incorrect element,
    # without conflating dataflow errors with accumulation rounding.
    rng = np.random.default_rng(args.seed)
    a = rng.integers(-1, 2, (m, k)).astype(np.float32) / 8
    b = rng.integers(-1, 2, (k, n)).astype(np.float32) / 8
    expected = (a @ b).astype("<f2")
    a, b = a.astype("<f2"), b.astype("<f2")
    regions = []

    def packed(parts):
        return b"".join(np.ascontiguousarray(part, dtype="<f2").tobytes() for part in parts)

    for cid in range(clusters):
        x, y = cid % nx, cid // nx
        if flat:
            pa = a.tobytes()
            pb = packed([b[:, cid * nt:(cid + 1) * nt]])
            gold = packed([expected[:, cid * nt:(cid + 1) * nt]])
        else:
            pa = packed([a[(i * ny + y) * mt:(i * ny + y + 1) * mt,
                           step * kt:(step + 1) * kt]
                         for i in range(mb) for step in range(steps)
                         if not summa or step % nx == x])
            pb = packed([b[step * kt:(step + 1) * kt,
                           (j * nx + x) * nt:(j * nx + x + 1) * nt]
                         for j in range(nb) for step in range(steps)
                         if not summa or step % ny == y])
            gold = packed([expected[(i * ny + y) * mt:(i * ny + y + 1) * mt,
                                    (j * nx + x) * nt:(j * nx + x + 1) * nt]
                           for i in range(mb) for j in range(nb)])
        assert len(pa) == acount * xb and len(pb) == bcount * wb
        assert len(gold) == mb * nb * zb
        data = bytearray(end)
        data[:len(pa)] = pa
        data[woff:woff + len(pb)] = pb
        # Poison C so missing stores or missing initialization fail validation.
        data[zoff:zoff + len(gold)] = b"\x00\x7e" * (len(gold) // 2)
        data[goldoff:goldoff + len(gold)] = gold
        regions.append((arch.dram3d_addr_base + cid * arch.dram3d_node_space, data))

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    elf64(out / "preload.elf", regions)
    macros = dict(M=m, N=n, K=k, MT=mt, NT=nt, KT=kt, MB=mb, NB=nb,
                  STEPS=steps, SUMMA=int(summa), BUFFERS=buffers, Z_BUFFERS=zbuffers,
                  X_BYTES=xb, W_BYTES=wb, Z_BYTES=zb,
                  W_OFFSET=woff, Z_OFFSET=zoff, GOLD_OFFSET=goldoff,
                  L1_X=lx, L1_W=lw, L1_Z=lz)
    (out / "gemm.h").write_text(
        "/* Generated by apps_dram3d/scripts/generate.py. */\n"
        '#include "flex_cluster_arch.h"\n' +
        "".join(f"#define D3_{name} {value}u\n" for name, value in macros.items()))
    # The SDK's legacy linker places .data in side HBM. Keep runtime data in
    # the reserved first 4 KiB of local TCDM for this stacked-only application.
    (out / "memory.ld").write_text(f"""OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {{
  L1 (rw) : ORIGIN = {arch.cluster_tcdm_base + 16:#x}, LENGTH = 0xff0
  CODE (rx) : ORIGIN = {arch.instruction_mem_base:#x}, LENGTH = {arch.instruction_mem_size:#x}
}}
PHDRS {{ code PT_LOAD; data PT_LOAD; }}
SECTIONS {{
  .text : {{ *(.init) *(.text*) *(.rodata*) }} > CODE :code
  .data : {{ __global_pointer$ = . + 0x800; *(.sdata*) *(.data*) *(.srodata*) }} > L1 :data
  .bss (NOLOAD) : {{ *(.sbss*) *(.bss*) *(COMMON) }} > L1 :data
  __l1_heap_start = ALIGN(16);
  __hbm_heap_start = 0;
}}
""")
    (out / "manifest.json").write_text(json.dumps(dict(
        case=args.case, shape=[m, n, k], dataflow="summa" if summa else "local",
        seed=args.seed, channels=clusters, tile=[mt, nt, kt],
        channel_bytes=end, tcdm_bytes=lz + zbuffers * zb - arch.cluster_tcdm_base,
        preload_bytes=sum(len(data) for _, data in regions),
        kernel_dram_bytes=mb * nb * steps *
            (xb * (ny if summa else clusters) + wb * (nx if summa else clusters)) + 2 * m * n,
        operations=2 * m * n * k, verified_elements=m * n), indent=2) + "\n")
    print(f"{args.case}: {m}x{n}x{k}, {'SUMMA' if summa else 'local'}, "
          f"{clusters} channels, {lz + zbuffers * zb - arch.cluster_tcdm_base} TCDM bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="big1024")
    parser.add_argument("--dataflow", choices=["auto", "summa", "local"], default="auto")
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19)
    generate(parser.parse_args())
