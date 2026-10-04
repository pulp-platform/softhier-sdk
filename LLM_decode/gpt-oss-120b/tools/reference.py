#!/usr/bin/env python3
"""Independent NumPy decoder reference and simulator output comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import struct

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np


def read_elf(path):
    regions = {}
    with path.open('rb') as f:
        header = f.read(64)
        phoff = struct.unpack_from('<Q', header, 32)[0]
        size, count = struct.unpack_from('<HH', header, 54)
        for i in range(count):
            f.seek(phoff + i * size)
            _, _, offset, _, addr, filesz, _, _ = struct.unpack('<IIQQQQQQ', f.read(56))
            regions[addr] = np.memmap(path, mode='r', offset=offset, dtype=np.uint8, shape=(filesz,))
    return regions


class Reference:
    def __init__(self, directory, batch):
        self.directory, self.batch = directory, batch
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        self.m = self.manifest['model']
        self.params, self.tensors = self.manifest['parameters'], self.manifest['tensors']
        self.inputs = read_elf(directory / f'input-b{batch}.elf')
        self.values = {}

    def weight(self, name):
        spec = self.params[name]
        data = np.memmap(self.directory / 'weights.elf', dtype='<f2', mode='r',
                         offset=spec['offset'], shape=(spec['size'] // 2,))
        if spec['packed']:
            k, n = spec['shape']; tk, tn = self.m['tile_k'], self.m['tile_n']
            nk, nn = (k + tk - 1) // tk, (n + tn - 1) // tn
            data = data.reshape(nn, nk, tk, tn).transpose(1, 2, 0, 3).reshape(nk * tk, nn * tn)[:k, :n]
        else:
            data = data.reshape(spec['shape'])
        return np.asarray(data, dtype=np.float32)

    def input(self, name, shape):
        data = self.inputs[self.tensors[name]['address']]
        return data.view('<f2').reshape(shape).copy()

    def norm(self, data, name):
        x = data.astype(np.float32)
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.m['rms_norm_eps']) * self.weight(name)).astype('<f2')

    def linear(self, x, weight_name, bias_name):
        w = self.weight(weight_name)
        # The accelerator keeps an FP16 partial sum between TK-sized launches.
        y = np.zeros((x.shape[0], w.shape[1]), dtype='<f2')
        tk = self.m['tile_k']
        for k in range(0, w.shape[0], tk):
            y = (y.astype(np.float32) + x[:, k:k + tk].astype(np.float32) @ w[k:k + tk]).astype('<f2')
        return (y.astype(np.float32) + self.weight(bias_name)).astype('<f2')

    def run(self):
        m, b = self.m, self.batch
        h, f, qh, kh, d, top, experts, ctx = (m[k] for k in ['hidden_size', 'intermediate_size',
            'num_attention_heads', 'num_key_value_heads', 'head_dim', 'experts_per_token',
            'num_experts', 'cached_tokens'])
        qdim, kvdim = qh * d, kh * d
        stride = self.manifest['cache_stride']
        out = self.values
        x = self.input('input', (b, h))
        out['norm1_out'] = self.norm(x, 'norm1')
        qkv = self.linear(out['norm1_out'], 'qkv_weight', 'qkv_bias')
        cs = self.weight('rope')
        for head in range(qh + kh):
            first = qkv[:, head * d:head * d + d // 2].astype(np.float32)
            second = qkv[:, head * d + d // 2:(head + 1) * d].astype(np.float32)
            qkv[:, head * d:head * d + d // 2] = (first * cs[:d // 2] - second * cs[d // 2:]).astype('<f2')
            qkv[:, head * d + d // 2:(head + 1) * d] = (second * cs[:d // 2] + first * cs[d // 2:]).astype('<f2')
        out['qkv'] = qkv
        q = qkv[:, :qdim].reshape(b, kh, qh // kh, d)
        keys = self.input('key_cache', (b, kh, d, stride))
        vals = self.input('value_cache', (b, kh, stride, d))
        keys[..., ctx] = qkv[:, qdim:qdim + kvdim].reshape(b, kh, d)
        vals[:, :, ctx] = qkv[:, qdim + kvdim:].reshape(b, kh, d)
        sinks = self.weight('sinks').reshape(kh, qh // kh, 1)
        attn = np.empty((b, kh, qh // kh, d), dtype='<f2')
        for token in range(b):
            for head in range(kh):
                scores = (q[token, head].astype(np.float32) @ keys[token, head, :, :ctx + 1].astype(np.float32)).astype('<f2').astype(np.float32) / np.sqrt(np.float32(d))
                maximum = np.maximum(scores.max(axis=-1, keepdims=True), sinks[head])
                probs = np.exp(scores - maximum)
                probs = (probs / (probs.sum(axis=-1, keepdims=True) + np.exp(sinks[head] - maximum))).astype('<f2')
                y = np.zeros((qh // kh, d), dtype='<f2')
                for k in range(0, ctx + 1, m['attention_tile']):
                    count = min(m['attention_tile'], ctx + 1 - k)
                    y = (y.astype(np.float32) + probs[:, k:k + count].astype(np.float32) @ vals[token, head, k:k + count].astype(np.float32)).astype('<f2')
                attn[token, head] = y
        out['attention'] = attn.reshape(b, qdim)
        out['projection'] = self.linear(out['attention'], 'out_weight', 'out_bias')
        out['residual'] = (x.astype(np.float32) + out['projection']).astype('<f2')
        out['norm2_out'] = self.norm(out['residual'], 'norm2')
        out['router_logits'] = self.linear(out['norm2_out'], 'router_weight', 'router_bias')
        ids = np.argsort(-out['router_logits'].astype(np.float32), axis=-1, kind='stable')[:, :top]
        scores = np.take_along_axis(out['router_logits'], ids, axis=-1).astype(np.float32)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = (probs / probs.sum(axis=-1, keepdims=True)).astype('<f2')
        out['routes'], out['route_weights'] = ids.astype('<u4'), probs
        dispatch, up, activation, down = [], [], [], []
        final = out['residual'].astype(np.float32).copy()
        for expert in range(experts):
            rows, ranks = np.where(ids == expert)
            if not len(rows): continue
            inputs = out['norm2_out'][rows]
            pairs = self.linear(inputs, f'up_weight_{expert}', f'up_bias_{expert}')
            gate = np.minimum(pairs[:, ::2].astype(np.float32), m['swiglu_limit'])
            linear = np.clip(pairs[:, 1::2].astype(np.float32), -m['swiglu_limit'], m['swiglu_limit'])
            active = (gate / (1 + np.exp(-np.float32(m['swiglu_alpha']) * gate)) * (linear + 1)).astype('<f2')
            result = self.linear(active, f'down_weight_{expert}', f'down_bias_{expert}')
            for i, token in enumerate(rows): final[token] += result[i].astype(np.float32) * np.float32(probs[token, ranks[i]])
            dispatch.append(inputs); up.append(pairs); activation.append(active); down.append(result)
        out['dispatch'] = np.concatenate(dispatch)
        out['gate_up'] = np.concatenate(up)
        out['activation'] = np.concatenate(activation)
        out['expert_output'] = np.concatenate(down)
        out['output'] = final.astype('<f2')
        return out


def read_dump(path, base):
    blocks = {}; address = None; words = []
    def save():
        if address is not None: blocks[address] = np.asarray(words, dtype='<u2').tobytes()
    for line in path.read_text().splitlines():
        if line.startswith('HBM offset == .0x'):
            save(); address = base + int(line.split('0x')[1].rstrip(':'), 16); words = []
        elif line.startswith('  0x'):
            words.append(int(line.strip(), 16))
    save()
    return blocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 64])
    p.add_argument('--atol', type=float, default=0.01)
    p.add_argument('--rtol', type=float, default=0.02)
    args = p.parse_args()
    directory = args.build_dir.resolve()
    for batch in args.batches:
        ref = Reference(directory, batch)
        expected = ref.run()
        # The SDK dump utility prints HBM-relative addresses.
        import runpy
        arch = runpy.run_path(ref.manifest['arch'])['FlexClusterArch']()
        blocks = read_dump(directory / f'batch-{batch}/dump_0', arch.hbm_start_base)
        results = {}; failed = []
        for name, gold in expected.items():
            address = ref.tensors[name]['address']
            if address not in blocks: continue
            data = bytearray()
            while len(data) < gold.nbytes:
                chunk = blocks.get(address + len(data))
                if chunk is None: raise AssertionError(f'Missing dump range for {name}')
                data += chunk
            assert len(data) == gold.nbytes, (name, len(data), gold.nbytes)
            actual = np.frombuffer(data, dtype=gold.dtype).reshape(gold.shape)
            if gold.dtype == np.dtype('<u4'):
                ok = np.array_equal(actual, gold); error = float(np.max(np.abs(actual.astype(float) - gold)))
            else:
                ok = np.allclose(actual, gold, atol=args.atol, rtol=args.rtol) and np.isfinite(actual).all()
                error = float(np.max(np.abs(actual.astype(float) - gold)))
            results[name] = dict(pass_check=bool(ok), max_abs_error=error, elements=gold.size)
            if not ok: failed.append(name)
        assert all(name in results for name in ['output', 'routes', 'route_weights'])
        (directory / f'batch-{batch}/validation.json').write_text(json.dumps(results, indent=2) + '\n')
        np.savez(directory / f'batch-{batch}/reference.npz', output=expected['output'],
                 routes=expected['routes'], route_weights=expected['route_weights'])
        print(f'batch {batch}: ' + json.dumps(results), flush=True)
        assert not failed, f'Numerical mismatch: {failed}'


if __name__ == '__main__':
    main()
