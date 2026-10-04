#!/usr/bin/env python3
"""Pack all FP16 layer weights and deterministic batch inputs into separate ELF64 files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import struct

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def align(n, a=128):
    return (n + a - 1) // a * a


def rope(model, positions):
    d = model['head_dim']
    idx = np.arange(d // 2, dtype=np.float32)
    freq = model['rope_theta'] ** (2 * idx / d)
    low = d / 2 * math.log(model['rope_original_context'] / (model['rope_beta_fast'] * 2 * math.pi)) / math.log(model['rope_theta'])
    high = d / 2 * math.log(model['rope_original_context'] / (model['rope_beta_slow'] * 2 * math.pi)) / math.log(model['rope_theta'])
    mask = 1 - np.clip((idx - low) / (high - low), 0, 1)
    inv = (1 - mask) / (model['rope_scaling_factor'] * freq) + mask / freq
    angle = np.asarray(positions, dtype=np.float32)[..., None] * inv
    concentration = 1 + 0.1 * math.log(model['rope_scaling_factor'])
    return (np.cos(angle) * concentration).astype('<f2'), (np.sin(angle) * concentration).astype('<f2')


class Elf:
    def __init__(self, path, count):
        self.file = path.open('wb')
        self.offset = align(64 + count * 56, 4096)
        self.file.seek(self.offset)
        self.segments = []

    def add(self, addr, array=None, memsz=None):
        if array is not None:
            array = np.ascontiguousarray(array)
            data = memoryview(array).cast('B')
            size = data.nbytes
            self.file.write(data)
        else:
            size = 0
        self.segments.append((addr, self.offset, size, size if memsz is None else memsz))
        offset = self.offset
        self.offset += size
        return offset, size

    def close(self):
        self.file.seek(0)
        self.file.write(b'\x7fELF\x02\x01\x01' + bytes(9) + struct.pack(
            '<HHIQQQIHHHHHH', 2, 243, 1, 0, 64, 0, 0, 64, 56, len(self.segments), 0, 0, 0))
        for addr, offset, size, memsz in self.segments:
            self.file.write(struct.pack('<IIQQQQQQ', 1, 6, offset, addr, addr, size, memsz, 1))
        self.file.close()


def pack(matrix, model):
    k, n = matrix.shape
    kt, nt = model['tile_k'], model['tile_n']
    padded = np.zeros((align(k, kt), align(n, nt)), dtype='<f2')
    padded[:k, :n] = matrix
    return padded.reshape(-1, kt, padded.shape[1] // nt, nt).transpose(2, 0, 1, 3).copy()


def layout(model, arch):
    h, f, qh, kh, d, e, b = (model[k] for k in ['hidden_size', 'intermediate_size',
        'num_attention_heads', 'num_key_value_heads', 'head_dim', 'num_experts', 'max_batch'])
    ctx = align(model['cached_tokens'] + 1, model['attention_tile'])
    bases = [arch.hbm_start_base, arch.hbm_start_base + arch.hbm_node_addr_space *
             (2 * arch.num_cluster_y + arch.num_cluster_x)]
    cursor = [0x10000000, 0x10000000]
    params = {}

    def add(name, shape, edge, matrix=False, kind='random', scale=0.01):
        size = np.prod(shape, dtype=np.int64) * 2
        if matrix: size = align(shape[0], model['tile_k']) * align(shape[1], model['tile_n']) * 2
        cursor[edge] = align(cursor[edge])
        params[name] = dict(address=bases[edge] + cursor[edge], shape=shape,
                            size=int(size), packed=matrix, kind=kind, scale=scale)
        cursor[edge] += int(size)
        assert cursor[edge] <= arch.hbm_node_addr_space, 'Weight region exceeds HBM mapping'

    add('norm1', [h], 0, kind='ones')
    add('norm2', [h], 0, kind='ones')
    add('qkv_weight', [h, (qh + 2 * kh) * d], 0, True, scale=0.01)
    add('qkv_bias', [(qh + 2 * kh) * d], 0)
    add('out_weight', [qh * d, h], 1, True, scale=0.01)
    add('out_bias', [h], 1)
    add('sinks', [qh], 0, scale=0.1)
    add('rope', [d], 0, kind='rope')
    # Structured router provides separated, reproducible top-k selections while still
    # executing the complete dense router GEMM. Expert matrices remain dense random data.
    add('router_weight', [h, e], 1, True, kind='router')
    add('router_bias', [e], 1, kind='zeros')
    for expert in range(e):
        edge = expert % 2
        add(f'up_weight_{expert}', [h, f * 2], edge, True)
        add(f'up_bias_{expert}', [f * 2], edge)
        add(f'down_weight_{expert}', [f, h], edge, True)
        add(f'down_bias_{expert}', [h], edge)

    tensors = {}
    cursor = [0x10000, 0x10000]

    def buf(name, shape, edge=0, dtype='float16'):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        cursor[edge] = align(cursor[edge])
        tensors[name] = dict(address=bases[edge] + cursor[edge], shape=shape, size=size, dtype=dtype)
        cursor[edge] += size
        assert cursor[edge] < 0x10000000, 'Activation workspace exceeds reserved HBM region'

    buf('control', [16], dtype='uint32')
    buf('input', [b, h])
    buf('norm1_out', [b, h])
    buf('qkv', [b, (qh + 2 * kh) * d])
    buf('key_cache', [b, kh, d, ctx])
    buf('value_cache', [b, kh, ctx, d], 1)
    buf('attention', [b, qh * d])
    buf('projection', [b, h])
    buf('residual', [b, h])
    buf('norm2_out', [b, h])
    buf('router_logits', [b, e])
    buf('routes', [b, model['experts_per_token']], dtype='uint32')
    buf('route_weights', [b, model['experts_per_token']])
    buf('metadata', [e * 2 + 1 + 2 * b * model['experts_per_token']], dtype='uint32')
    buf('dispatch', [b * model['experts_per_token'], h])
    buf('gate_up', [b * model['experts_per_token'], 2 * f], 1)
    buf('activation', [b * model['experts_per_token'], f], 1)
    buf('expert_output', [b * model['experts_per_token'], h], 1)
    buf('output', [b, h])
    return params, tensors, ctx


def weights(path, model, params):
    elf = Elf(path, len(params))
    for index, (name, spec) in enumerate(params.items()):
        shape = spec['shape']
        if spec['kind'] == 'ones': data = np.ones(shape, dtype='<f2')
        elif spec['kind'] == 'zeros': data = np.zeros(shape, dtype='<f2')
        elif spec['kind'] == 'rope':
            c, s = rope(model, model['cached_tokens'])
            data = np.concatenate([c, s])
        elif spec['kind'] == 'router':
            data = np.zeros(shape, dtype='<f2')
            for expert in range(model['num_experts']): data[expert, expert] = 0.25
        else:
            # Each tensor seed is independent of batch size and generation order.
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little') ^ model['seed']
            data = (np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * spec['scale']).astype('<f2')
        if spec['packed']: data = pack(data, model)
        spec['offset'], size = elf.add(spec['address'], data)
        assert size == spec['size'], name
        spec['sha256'] = hashlib.sha256(memoryview(data).cast('B')).hexdigest()
        if index % 32 == 0: print(f'weights: {index + 1}/{len(params)}', flush=True)
    elf.close()


def batch_input(path, model, tensors, batch, stride):
    h, kh, d, ctx, top = (model[k] for k in ['hidden_size', 'num_key_value_heads',
                                          'head_dim', 'cached_tokens', 'experts_per_token'])
    elf = Elf(path, 4)
    control = np.array([0x47505431, batch, ctx, int(h <= 64)] + [0] * 12, dtype='<u4')
    elf.add(tensors['control']['address'], control)
    data = np.random.default_rng(model['seed'] + 1).uniform(-0.25, 0.25, (batch, h)).astype('<f2')
    data[:, :model['num_experts']] = 0
    for token in range(batch):
        for rank in range(top):
            expert = (token * 37 + rank * 11) % model['num_experts']
            data[token, expert] = 4.4 - 0.2 * rank
    elf.add(tensors['input']['address'], data)
    # Per-token RNG streams preserve the same input/cache prefix for every batch.
    keys = np.zeros((batch, kh, d, stride), dtype='<f2')
    values = np.zeros((batch, kh, stride, d), dtype='<f2')
    cos, sin = rope(model, np.arange(ctx))
    for token in range(batch):
        rng = np.random.default_rng(model['seed'] + 1000 + token)
        raw = (rng.standard_normal((kh, ctx, d), dtype=np.float32) * 0.25).astype('<f2').astype(np.float32)
        a, z = raw[..., :d // 2], raw[..., d // 2:]
        rotated = np.concatenate([a * cos - z * sin, z * cos + a * sin], axis=-1).astype('<f2')
        keys[token, :, :, :ctx] = rotated.transpose(0, 2, 1)
        values[token, :, :ctx] = (rng.standard_normal((kh, ctx, d), dtype=np.float32) * 0.25).astype('<f2')
    elf.add(tensors['key_cache']['address'], keys)
    elf.add(tensors['value_cache']['address'], values)
    elf.close()


def write_header(path, model, params, tensors, stride):
    lines = ['#pragma once', '#include <stdint.h>']
    for name, value in model.items():
        if isinstance(value, int): lines.append(f'#define MODEL_{name.upper()} {value}')
        elif isinstance(value, float): lines.append(f'#define MODEL_{name.upper()} {value}f')
    lines += [f'#define CACHE_STRIDE {stride}',
              f'#define QKV_SIZE {(model["num_attention_heads"] + 2 * model["num_key_value_heads"]) * model["head_dim"]}',
              '#define Q_SIZE (MODEL_NUM_ATTENTION_HEADS * MODEL_HEAD_DIM)',
              '#define KV_SIZE (MODEL_NUM_KEY_VALUE_HEADS * MODEL_HEAD_DIM)',
              '#define Q_GROUP (MODEL_NUM_ATTENTION_HEADS / MODEL_NUM_KEY_VALUE_HEADS)']
    for name, spec in {**params, **tensors}.items():
        if name.rsplit('_', 1)[-1].isdigit(): continue
        lines.append(f'#define HBM_{name.upper()} UINT64_C(0x{spec["address"]:x})')
    for group in ['up_weight', 'up_bias', 'down_weight', 'down_bias']:
        lines.append(f'static const uint64_t HBM_{group.upper()}[MODEL_NUM_EXPERTS] = {{' +
                     ','.join(f'UINT64_C(0x{params[f"{group}_{i}"]["address"]:x})' for i in range(model['num_experts'])) + '};')
    path.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, default=HERE / 'config/model.json')
    p.add_argument('--arch', type=Path, default=HERE / 'config/arch.py')
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 64])
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--reuse-weights', action='store_true')
    args = p.parse_args()
    out = args.build_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.model.read_text())
    if args.smoke:
        model.update(hidden_size=64, intermediate_size=64, num_attention_heads=8,
                     num_key_value_heads=2, head_dim=8, num_experts=8, experts_per_token=2,
                     cached_tokens=16, max_batch=4, tile_k=64, tile_n=32)
    assert all(0 < b <= model['max_batch'] for b in args.batches)
    arch = runpy.run_path(str(args.arch.resolve()))['FlexClusterArch']()
    params, tensors, stride = layout(model, arch)
    manifest_path = out / 'manifest.json'
    if args.reuse_weights:
        previous = json.loads(manifest_path.read_text())
        assert previous['model'] == model
        assert {k: v['address'] for k, v in previous['parameters'].items()} == {k: v['address'] for k, v in params.items()}
        params = previous['parameters']
    else:
        weights(out / 'weights.elf', model, params)
    for batch in args.batches:
        batch_input(out / f'input-b{batch}.elf', model, tensors, batch, stride)
    write_header(out / 'model.h', model, params, tensors, stride)
    manifest = dict(model=model, arch=str(args.arch.resolve()), parameters=params, tensors=tensors,
                    cache_stride=stride, batches=args.batches,
                    weights_bytes=sum(s['size'] for s in params.values()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Generated {out}; weights {manifest["weights_bytes"]:,} bytes', flush=True)


if __name__ == '__main__':
    main()
