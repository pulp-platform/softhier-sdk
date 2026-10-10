# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Synthetic FP16 transformer block and NumPy reference (no checkpoint required).

Architecture: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
This is an FP16 adaptation, not MXFP4/BF16 checkpoint inference.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

MT, NT, KT, BC = 16, 64, 128, 128
CASES = {f"decode_b{b}_kv{k}": (b, 1, k)
         for b in (1, 8, 16) for k in (1024, 2048, 4096)}
CASES.update({f"prefill_s{s}": (1, s, 0) for s in (64, 128)})


@dataclass(frozen=True)
class Model:
    hidden: int = 2880
    intermediate: int = 2880
    heads: int = 64
    kv_heads: int = 8
    head_dim: int = 64
    experts: int = 128
    topk: int = 4

    @property
    def qdim(self):
        return self.heads * self.head_dim

    @property
    def kvdim(self):
        return self.kv_heads * self.head_dim

    @property
    def qkv(self):
        return self.qdim + 2 * self.kvdim

    def dictionary(self):
        return asdict(self)


def half(x):
    return np.asarray(x, dtype="<f2")


def matmul(a, b):
    """Model the FP16 accumulator spill at each software K=128 boundary.

    RedMule accumulates FP32 within a tile; the default hardware reduction
    tile is at least 128. BLAS association can differ slightly from its scalar
    host loop, so validation uses an explicit absolute/relative tolerance.
    """
    c = np.zeros((len(a), b.shape[1]), np.float16)
    for k in range(0, a.shape[1], KT):
        c = half(c.astype(np.float32) +
                 a[:, k:k + KT].astype(np.float32) @ b[k:k + KT].astype(np.float32))
    return c


class Weights:
    """Independent streams make every tensor reproducible without a 6 GB cache."""
    def __init__(self, model, seed):
        self.m, self.seed = model, seed

    def values(self, tag, shape, scale, expert=0):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, tag, expert]))
        return half(rng.integers(-4, 5, shape, dtype=np.int16).astype(np.float32) * scale)

    def matrix(self, kind, expert=0):
        m = self.m
        shapes = {"qkv": (m.hidden, m.qkv), "out": (m.qdim, m.hidden),
                  "router": (m.hidden, m.experts),
                  "up": (m.hidden, 2 * m.intermediate),
                  "down": (m.intermediate, m.hidden)}
        tags = {"qkv": 1, "out": 2, "router": 3, "up": 4, "down": 5}
        k, n = shapes[kind]
        # Power-of-two values keep products well behaved in FP32 accumulation.
        scale = 2.0 ** -math.ceil(math.log2(math.sqrt(k) * 4))
        return self.values(tags[kind], (k, n), scale, expert)

    def bias(self, kind, expert=0):
        m = self.m
        sizes = {"qkv": m.qkv, "out": m.hidden, "router": m.experts,
                 "up": 2 * m.intermediate, "down": m.hidden}
        tags = {"qkv": 11, "out": 12, "router": 13, "up": 14, "down": 15}
        return self.values(tags[kind], (sizes[kind],), 1 / 128, expert)

    def norm(self, second=False):
        return half(1 + self.values(21 + int(second), (self.m.hidden,), 1 / 64).astype(np.float32))

    def sinks(self):
        return self.values(23, (self.m.heads,), 1 / 8)


def rmsnorm(x, weight):
    x = x.astype(np.float32)
    # Sequential sum agrees with the scalar kernel more closely than pairwise
    # numpy.sum, while still computing all tokens in parallel.
    squares = x * x
    total = np.cumsum(squares, axis=1, dtype=np.float32)[:, -1:]
    return half(x * (np.float32(1) / np.sqrt(total / x.shape[1] + np.float32(1e-5))) * weight)


def rope_table(m, positions):
    d = m.head_dim // 2
    freq = np.float32(150000) ** (np.arange(d, dtype=np.float32) * np.float32(2 / m.head_dim))
    low = d * math.log(4096 / (32 * 2 * math.pi)) / math.log(150000)
    high = d * math.log(4096 / (2 * math.pi)) / math.log(150000)
    mask = 1 - np.clip((np.arange(d, dtype=np.float32) - low) / (high - low), 0, 1)
    inv = (1 - mask) / (32 * freq) + mask / freq
    angle = np.asarray(positions, np.float32)[:, None] * inv
    concentration = np.float32(1 + 0.1 * math.log(32))
    # Rows contain cos[0:d], sin[0:d]; split-half RoPE, not interleaved RoPE.
    return half(np.concatenate((np.cos(angle) * concentration,
                                np.sin(angle) * concentration), axis=1))


def apply_rope(x, table):
    d = x.shape[-1] // 2
    a, b = x[..., :d].astype(np.float32), x[..., d:].astype(np.float32)
    co, si = table[:, None, :d].astype(np.float32), table[:, None, d:].astype(np.float32)
    return half(np.concatenate((a * co - b * si, b * co + a * si), axis=-1))


def routing(logits, topk):
    # Stable sort gives the C implementation's smaller-index tie break.
    ids = np.argsort(-logits.astype(np.float32), axis=1, kind="stable")[:, :topk]
    selected = np.take_along_axis(logits.astype(np.float32), ids, axis=1)
    scores = np.exp(selected - selected[:, :1])
    scores /= np.sum(scores, axis=1, keepdims=True, dtype=np.float32)
    return ids.astype("<u4"), scores.astype("<f4")


def swiglu(x):
    x = x.astype(np.float32)
    gate = np.minimum(x[:, ::2], np.float32(7))
    linear = np.clip(x[:, 1::2], -7, 7)
    return half((gate / (1 + np.exp(-np.float32(1.702) * gate))) * (linear + 1))


def attention(q, k, v, sinks, window=0, past=0):
    """Causal GQA with sink denominator and FP32 online softmax state.

    q: [batch, sequence, KV heads, Q heads/KV head, D].
    k/v include the existing prefix and all newly appended tokens.
    """
    bsz, seq, nh, group, dim = q.shape
    result = np.empty_like(q)
    for b in range(bsz):
        for h in range(nh):
            for t in range(seq):
                end = past + t + 1
                start = max(0, end - window) if window else 0
                maximum = sinks[h * group:(h + 1) * group].astype(np.float32).copy()
                denom = np.ones(group, np.float32)
                acc = np.zeros((group, dim), np.float32)
                for j in range(start, end, BC):
                    stop = min(end, j + BC)
                    scores = matmul(q[b, t, h], k[b, j:stop, h].T).astype(np.float32)
                    scores *= np.float32(1 / math.sqrt(dim))
                    newmax = np.maximum(maximum, np.max(scores, axis=1))
                    alpha = np.exp(maximum - newmax)
                    p = np.exp(scores - newmax[:, None])
                    # The P@V RedMule input/output are FP16, online state FP32.
                    pv = matmul(half(p), v[b, j:stop, h]).astype(np.float32)
                    acc = acc * alpha[:, None] + pv
                    denom = denom * alpha + np.sum(p, axis=1, dtype=np.float32)
                    maximum = newmax
                result[b, t, h] = half(acc / denom[:, None])
    return result.reshape(bsz * seq, nh * group * dim)


def reference(m, batch, seq, past, layer, seed, progress=lambda _: None):
    w = Weights(m, seed)
    tokens = batch * seq
    x = w.values(31, (tokens, m.hidden), 1 / 8)
    # Prefix K is already rotated, as a real layer KV cache would be.
    k = np.zeros((batch, past + seq, m.kv_heads, m.head_dim), dtype="<f2")
    v = np.zeros_like(k)
    k[:, :past] = w.values(32, (batch, past, m.kv_heads, m.head_dim), 1 / 8)
    v[:, :past] = w.values(33, (batch, past, m.kv_heads, m.head_dim), 1 / 8)
    gold = {"norm1": rmsnorm(x, w.norm())}
    progress("QKV projection")
    qkv = half(matmul(gold["norm1"], w.matrix("qkv")).astype(np.float32) + w.bias("qkv"))
    table = rope_table(m, np.tile(np.arange(past, past + seq), batch))
    q = apply_rope(qkv[:, :m.qdim].reshape(tokens, m.heads, m.head_dim), table)
    knew = apply_rope(qkv[:, m.qdim:m.qdim + m.kvdim].reshape(tokens, m.kv_heads, m.head_dim), table)
    vnew = qkv[:, m.qdim + m.kvdim:].reshape(batch, seq, m.kv_heads, m.head_dim)
    qkv[:, :m.qdim] = q.reshape(tokens, m.qdim)
    qkv[:, m.qdim:m.qdim + m.kvdim] = knew.reshape(tokens, m.kvdim)
    gold["qkv"] = qkv
    k[:, past:] = knew.reshape(batch, seq, m.kv_heads, m.head_dim)
    v[:, past:] = vnew
    progress("causal attention")
    gold["attn"] = attention(q.reshape(batch, seq, m.kv_heads, m.heads // m.kv_heads, m.head_dim),
                             k, v, w.sinks(), 128 if layer % 2 == 0 else 0, past)
    projected = half(matmul(gold["attn"], w.matrix("out")).astype(np.float32) + w.bias("out"))
    gold["projected"] = projected
    gold["residual"] = half(x.astype(np.float32) + projected.astype(np.float32))
    gold["norm2"] = rmsnorm(gold["residual"], w.norm(True))
    gold["router"] = half(matmul(gold["norm2"], w.matrix("router")).astype(np.float32) + w.bias("router"))
    ids, scores = routing(gold["router"], m.topk)
    experts = np.unique(ids).tolist()
    partial = np.empty((tokens, m.topk, m.hidden), dtype="<f2")
    for i, expert in enumerate(experts):
        progress(f"expert {i + 1}/{len(experts)} (id {expert})")
        rows, slots = np.where(ids == expert)
        up = half(matmul(gold["norm2"][rows], w.matrix("up", expert)).astype(np.float32) + w.bias("up", expert))
        down = half(matmul(swiglu(up), w.matrix("down", expert)).astype(np.float32) + w.bias("down", expert))
        partial[rows, slots] = down
    gold["partial"] = partial.reshape(tokens, -1)
    combined = np.zeros((tokens, m.hidden), np.float32)
    for i in range(m.topk):
        combined += partial[:, i].astype(np.float32) * scores[:, i:i + 1]
    gold["output"] = half(gold["residual"].astype(np.float32) + combined)
    return dict(x=x, gold=gold, ids=ids, scores=scores, rope=table, k=k, v=v,
                active_experts=experts)
