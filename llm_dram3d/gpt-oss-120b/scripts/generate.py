#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Plan the distributed layer and stream a sparse ELF64 preload plus gold data."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np

from model import BC, CASES, KT, MT, NT, Model, Weights, reference


def ceildiv(a, b):
    return (a + b - 1) // b


def align(n):
    return ceildiv(n, 128) * 128


def load_arch(path):
    spec = importlib.util.spec_from_file_location("llm_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlexClusterArch()


def layout(m, batch, seq, past, arch):
    nc = arch.num_cluster_x * arch.num_cluster_y
    if (arch.num_cluster_x, arch.num_cluster_y) != (4, 4):
        raise ValueError("This initial mapping requires the 4x4 cluster preset")
    if not arch.dram3d_enable or any(arch.hbm_chan_placement):
        raise ValueError("Use the stacked-only apps_dram3d architecture")
    if 0 not in arch.spatz_attaced_core_list:
        raise ValueError("The vector kernels require Spatz on core 0")
    if arch.dram3d_type != "hbm2-example.json":
        raise ValueError("Channel capacity is checked for hbm2-example.json (2 GiB)")
    reduction_tile = arch.cluster_tcdm_bank_width // 8 * arch.cluster_tcdm_bank_nb // 2
    if reduction_tile < KT or arch.redmule_elem_size != 2:
        raise ValueError("Reference requires an FP16 RedMule reduction tile >= 128")
    offsets, cursor = {}, 0

    def reserve(name, size):
        nonlocal cursor
        offsets[name] = cursor
        cursor = align(cursor + size)

    reserve("norm1_weight", m.hidden * 2)
    reserve("norm2_weight", m.hidden * 2)
    reserve("sinks", m.heads * 2)
    matrices = {}
    for name, k, n in (("qkv", m.hidden, m.qkv), ("out", m.qdim, m.hidden),
                       ("router", m.hidden, m.experts)):
        tiles = ceildiv(ceildiv(n, NT), nc)
        reserve("w_" + name, tiles * ceildiv(k, KT) * KT * NT * 2)
        reserve("b_" + name, tiles * NT * 2)
        matrices[name] = [k, n]
    expert_start = cursor
    reserve("w_up", ceildiv(2 * m.intermediate, NT) * ceildiv(m.hidden, KT) * KT * NT * 2)
    reserve("b_up", ceildiv(2 * m.intermediate, NT) * NT * 2)
    reserve("w_down", ceildiv(m.hidden, NT) * ceildiv(m.intermediate, KT) * KT * NT * 2)
    reserve("b_down", ceildiv(m.hidden, NT) * NT * 2)
    stride = cursor - expert_start
    cursor = expert_start + ceildiv(m.experts, nc) * stride
    weight_end = cursor
    rows = ceildiv(batch * seq, nc)
    fields = dict(x=m.hidden, norm1=m.hidden, qkv=m.qkv, attn=m.qdim,
                  projected=m.hidden, residual=m.hidden, norm2=m.hidden,
                  router=m.experts, partial=m.topk * m.hidden, output=m.hidden)
    for name, width in fields.items():
        reserve(name, rows * width * 2)
        if name != "x":
            reserve("gold_" + name, rows * width * 2)
    reserve("routes", rows * m.topk * 8)
    reserve("gold_routes", rows * m.topk * 8)
    reserve("rope", batch * seq * m.head_dim * 2)
    # Each (batch, KV head) has a single home; job % nc is its channel.
    cache_stride = align((past + seq) * m.head_dim * 2)
    reserve("cache_k", ceildiv(batch * m.kv_heads, nc) * cache_stride)
    reserve("cache_v", ceildiv(batch * m.kv_heads, nc) * cache_stride)
    # Reused by successive experts on a channel; at most MT routed tokens live.
    reserve("up", MT * 2 * m.intermediate * 2)
    reserve("activation", MT * m.intermediate * 2)
    if cursor > min(arch.dram3d_node_space, 2**31):
        raise ValueError("Layer exceeds the DRAM channel capacity or its aperture")
    max_k = max(m.hidden, m.intermediate, m.qdim)
    group = m.heads // m.kv_heads
    attention_bytes = 2 * (group * m.head_dim + 3 * BC * m.head_dim +
                           group * BC + group * m.head_dim) + 4 * (group * m.head_dim + 3 * group)
    l1, lc = {}, arch.cluster_tcdm_base + 0x1000
    for name, size in (("a", max(MT * ceildiv(max_k, KT) * KT * 2, attention_bytes)),
                       ("x", MT * KT * 2), ("w", KT * NT * 2),
                       ("c", MT * NT * 2), ("bias", NT * 2),
                       ("tmp", max(m.qkv * 4, m.topk * m.hidden * 2, 32768)),
                       ("routes", batch * seq * m.topk * 8)):
        l1[name] = lc
        lc = align(lc + size)
    if lc > arch.cluster_tcdm_base + arch.cluster_tcdm_size:
        raise ValueError("Local buffers exceed TCDM")
    return dict(offsets=offsets, matrices=matrices, fields=fields,
                expert_stride=stride, cache_stride=cache_stride,
                channel_bytes=cursor, weight_bytes_per_channel=weight_end,
                tcdm_bytes=lc - arch.cluster_tcdm_base, l1=l1, channels=nc)


class ElfWriter:
    """Stream segments; never assemble a multi-gigabyte layer in a bytearray."""
    def __init__(self, path):
        self.file = path.open("wb")
        self.headers = []
        self.file.seek(64 + 2048 * 56)
        self.payload = 0

    def add(self, address, array):
        array = np.ascontiguousarray(array)
        if not array.nbytes:
            return
        if len(self.headers) >= 2048:
            raise ValueError("Too many ELF segments")
        offset = self.file.tell()
        self.headers.append(struct.pack("<IIQQQQQQ", 1, 6, offset, address, address,
                                        array.nbytes, array.nbytes, 1))
        array.tofile(self.file)
        self.payload += array.nbytes

    def close(self):
        self.file.seek(0)
        self.file.write(b"\x7fELF\x02\x01\x01" + bytes(9) + struct.pack(
            "<HHIQQQIHHHHHH", 2, 243, 1, 0, 64, 0, 0, 64, 56, len(self.headers), 0, 0, 0))
        self.file.write(b"".join(self.headers))
        self.file.close()


def pack_matrix(matrix, tiles):
    k, n = matrix.shape
    blocks = np.zeros((len(tiles), ceildiv(k, KT), KT, NT), dtype="<f2")
    for local, j in enumerate(tiles):
        width = min(NT, n - j * NT)
        for kb in range(ceildiv(k, KT)):
            height = min(KT, k - kb * KT)
            blocks[local, kb, :height, :width] = matrix[kb * KT:kb * KT + height, j * NT:j * NT + width]
    return blocks


def generate(args):
    m = Model()
    batch, seq, past = CASES[args.case]
    if not 0 <= args.layer < 36:
        raise ValueError("Layer index must be in [0, 35]")
    arch = load_arch(args.arch)
    plan = layout(m, batch, seq, past, arch)
    nc, offsets = plan["channels"], plan["offsets"]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(case=args.case, model=m.dictionary(),
                    batch=batch, sequence=seq, past_kv=past, layer=args.layer,
                    attention="sliding128" if args.layer % 2 == 0 else "full",
                    seed=args.seed, arch=str(args.arch.resolve()), **plan)
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return
    # A failed regeneration must not leave a previously passing result.
    for name in ("manifest.json", "results.json"):
        (out / name).unlink(missing_ok=True)
    print(f"{args.case}: building the numerical reference", flush=True)
    data = reference(m, batch, seq, past, args.layer, args.seed,
                     lambda msg: print("  " + msg, flush=True))
    w = Weights(m, args.seed)
    loaded = list(range(m.experts)) if args.all_experts else data["active_experts"]
    writer = ElfWriter(out / "preload.elf.tmp")

    def add(cid, name, array, extra=0):
        address = arch.dram3d_addr_base + cid * arch.dram3d_node_space + offsets[name] + extra
        writer.add(address, array)

    print("  packing dense weights and token buffers", flush=True)
    for kind in ("qkv", "out", "router"):
        matrix, bias = w.matrix(kind), w.bias(kind)
        for cid in range(nc):
            tiles = list(range(cid, ceildiv(matrix.shape[1], NT), nc))
            add(cid, "w_" + kind, pack_matrix(matrix, tiles))
            padded_bias = np.zeros((len(tiles), NT), dtype="<f2")
            for local, tile in enumerate(tiles):
                valid = min(NT, len(bias) - tile * NT)
                padded_bias[local, :valid] = bias[tile * NT:tile * NT + valid]
            add(cid, "b_" + kind, padded_bias)
    for expert in loaded:
        cid, extra = expert % nc, expert // nc * plan["expert_stride"]
        for kind in ("up", "down"):
            matrix, bias = w.matrix(kind, expert), w.bias(kind, expert)
            add(cid, "w_" + kind, pack_matrix(matrix, list(range(ceildiv(matrix.shape[1], NT)))), extra)
            padded_bias = np.zeros(ceildiv(len(bias), NT) * NT, dtype="<f2")
            padded_bias[:len(bias)] = bias
            add(cid, "b_" + kind, padded_bias, extra)
    for cid in range(nc):
        add(cid, "norm1_weight", w.norm())
        add(cid, "norm2_weight", w.norm(True))
        add(cid, "sinks", w.sinks())
        add(cid, "rope", data["rope"])
        add(cid, "x", data["x"][cid::nc])
        for name, expected in data["gold"].items():
            add(cid, name, np.full_like(expected[cid::nc], np.nan))
            add(cid, "gold_" + name, expected[cid::nc])
        count = len(data["ids"][cid::nc])
        routes = np.empty((count, 2, m.topk), dtype="<u4")
        routes[:, 0] = data["ids"][cid::nc]
        routes[:, 1] = data["scores"][cid::nc].view("<u4")
        add(cid, "gold_routes", routes)
        add(cid, "routes", np.full_like(routes, 0xffffffff))
    for b in range(batch):
        for h in range(m.kv_heads):
            job = b * m.kv_heads + h
            for kind in ("k", "v"):
                cache = data[kind][b, :, h].copy()
                cache[past:] = np.nan  # Kernel must append the projected K/V.
                add(job % nc, "cache_" + kind, cache, job // nc * plan["cache_stride"])
    writer.close()
    (out / "preload.elf.tmp").replace(out / "preload.elf")
    macros = dict(H=m.hidden, I=m.intermediate, HEADS=m.heads, KV_HEADS=m.kv_heads,
                  D=m.head_dim, GROUP=m.heads // m.kv_heads, EXPERTS=m.experts,
                  TOPK=m.topk, QDIM=m.qdim, KVDIM=m.kvdim, QKV=m.qkv,
                  BATCH=batch, SEQ=seq, PAST=past, TOKENS=batch * seq,
                  WINDOW=128 if args.layer % 2 == 0 else 0, LAYER=args.layer,
                  MT=MT, NT=NT, KT=KT, BC=BC,
                  EXPERT_STRIDE=plan["expert_stride"], CACHE_STRIDE=plan["cache_stride"])
    macros.update({"OFF_" + key.upper(): value for key, value in offsets.items()})
    macros.update({"L1_" + key.upper(): value for key, value in plan["l1"].items()})
    masks = [sum(1 << (e % 32) for e in loaded if e // 32 == block)
             for block in range(ceildiv(m.experts, 32))]
    (out / "layer.h").write_text(
        '/* Generated; addresses of stacked channels stay uint64_t. */\n'
        '#include <stdint.h>\n#include "flex_cluster_arch.h"\n' +
        f"#define L_SM_SCALE {1 / math.sqrt(m.head_dim):.12g}f\n" +
        "".join(f"#define L_{name} {value}u\n" for name, value in macros.items()) +
        "static const uint32_t loaded_experts[] = {" + ",".join(f"{x}u" for x in masks) + "};\n")
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
    checked = batch * seq * sum(a.shape[1] for a in data["gold"].values())
    # Also check appended K and V, plus both integer and FP32 route entries.
    checked += batch * seq * (2 * m.kvdim + 2 * m.topk)
    manifest.update(active_experts=data["active_experts"], loaded_experts=loaded,
                    preload_bytes=writer.payload, verified_elements=checked,
                    validation_atol=0.003, validation_rtol=0.02,
                    precision="FP16 matrices; FP32 vector reductions and nonlinear functions")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  ready: {writer.payload:,} preload bytes, {plan['channel_bytes']:,} bytes/channel, "
          f"{plan['tcdm_bytes']:,} TCDM bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(CASES), default="decode_b1_kv1024")
    p.add_argument("--arch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--all-experts", action="store_true", help="Preload inactive experts too (default: reserve only)")
    p.add_argument("--plan-only", action="store_true", help="Check capacities and print the mapping without generating weights")
    generate(p.parse_args())
