# GPT-OSS-120B layer on stacked DRAM

This application executes one complete transformer block with deterministic
synthetic FP16 weights and a NumPy numerical reference. It uses the existing
`apps_dram3d/config/arch.py`: 4×4 clusters, one stacked HBM2 channel per cluster,
384 KiB TCDM, and no side HBM. Generated artifacts stay in this application's
ignored `build/` directory.

The dimensions follow the [official model configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json):
hidden size 2880, intermediate size 2880, 64 query heads, 8 KV heads, head dimension
64, and 128 experts with four selected per token. The block follows the
[official layer implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py):
RMSNorm → biased QKV → YaRN RoPE → causal GQA with attention sinks → biased output
projection → residual → RMSNorm → biased router/top-k/softmax → biased expert
gate/up, clipped interleaved SwiGLU, biased down → weighted combine → residual.
It excludes embeddings, the final model normalization, and the vocabulary head.

`LAYER=1` selects a full-attention layer. Even layer indices select the alternating
128-token sliding window; valid indices are 0–35. `KV` means **cached past tokens**:
decode appends one token and attends to `KV+1` positions. Prefill starts with an
empty cache, uses batch one, and produces all 64 or 128 output tokens. Sequences
in a decode batch have independent caches.

This workload uses generated weights to check architecture execution; it does
not load pretrained checkpoints. FP32 vector reductions and nonlinear operations
feed FP16 matrices. The reference accounts for the software K-tile accumulator spills
and FP16 attention probabilities; results are checked with `atol=0.003`,
`rtol=0.02`, and exact expert indices. Each intermediate tensor, each expert
contribution, every output element, route weights, and appended K/V are checked.
Outputs start as NaNs to expose missing writes.

## Running

Use the SDK's RISC-V toolchain and an already built `soft_hier_old` simulator with
DRAMSys storage enabled. The host needs Python 3.9+ and NumPy. Set the usual GVSoC
and DRAMSys environment as for `apps_dram3d`; the makefiles only build software.

From this directory:

```sh
make plan CASE=decode_b16_kv4096    # Show addresses/capacity without generating weights
make -C decode run CASE=decode_b1_kv1024
make -C decode suite               # B=1,8,16 × past KV=1024,2048,4096
make -C prefill run CASE=prefill_s64
make -C prefill suite              # B=1 × prompt length=64,128
make suite                         # All eleven full-size cases
```

All eleven presets use the full model dimensions. Decode case names follow
`decode_b{1,8,16}_kv{1024,2048,4096}`; prefill cases are `prefill_s64` and
`prefill_s128`. Defaults are `CASE=decode_b1_kv1024`, `SEED=19`, `LAYER=1`, and
`TIMEOUT=3600` seconds. Override `ARCH` or `GVSOC` to select the configuration or
simulator path. `OUT` defaults to `build/<case>-l<layer>-s<seed>`.

After generation, `make compile` and `make simulate` reuse the preload. Keep
the same `CASE`, `LAYER`, and `SEED`, or pass the generated directory as `OUT`.

The generated `manifest.json` gives tensor offsets, dimensions, capacity,
preloaded experts, and validation counts. `simulation.log` contains diagnostics;
`results.json` is written only after a verified pass, with total and per-stage
cycles at 1 GHz. Timing includes layer DMA, computation, and synchronization;
preload and numerical checking are excluded. Simulator exit status alone is not
accepted as a pass.
The legacy core exposes a 32-bit `cycle` CSR. Each stage uses a wrapping unsigned
difference (each stage must take fewer than 2³² cycles), and the total is accumulated
in 64 bits. No compressed-instruction libm/libgcc routines are called.

## Mapping and execution

| Object | Placement |
|---|---|
| Channel address | `0x10000000000 + cid * 0xc0000000 + offset` |
| Dense projection weights | 64-column tiles, tile `j` on channel `j % 16` |
| Expert weights | Expert `e` on channel `e % 16`; eight experts per channel |
| Token rows, layer I/O, intermediates | Flattened token `t` on channel `t % 16` |
| KV cache | Entire `(batch, KV-head)` on channel `(batch*8 + head) % 16` |
| Expert scratch | Reused up/gate and activation buffers in the expert owner's channel |
| Top-k expert outputs | Four distinct slots in the token owner's channel |
| Local working buffers | Explicit bounded arena in TCDM; zero-memory DMA clears |

Every DRAM address is 64-bit through the DMA interface. Allocation is checked
against the HBM2 model's **2 GiB capacity**, as well as the 3 GiB channel aperture.
All 128 expert slots are reserved. By default the untimed preload includes the
experts selected by the generated reference inputs; the device still computes
its own routing and rejects any selection whose weights are absent. Set
`ALL_EXPERTS=1` to preload inactive experts too. This changes initialization/file
size, not kernel traffic or the reserved memory layout. Large prefill cases can
still generate several GB and take substantial host simulation time.
Increase `TIMEOUT` for long full-size prefill simulations if necessary.

RedMule computes dense/expert projections and attention QK/PV products. Tiling is
M≤16, N=64, K=128 for projections, with explicit zero padding. Attention streams
128 cache positions at a time, uses the transpose engine for K, and maintains an
FP32 online softmax denominator and output accumulator, including each head's
sink logit in the denominator only.

Spatz vector code handles RMSNorm, split-half RoPE, softmax reductions and
exponentials, router maximum reductions, SwiGLU, biases, and residual/expert
combination. It follows the SDK's `implementation/sw/RMSNorm`, `Activation`, and
`FlatAttention` instruction patterns, including custom `vfexp.vv`. The first
schedule uses core 0's Spatz and RedMule in each active cluster. Other cores join
barriers. Scalar code handles DMA/control, one-value softmax state updates, and
the index scan after each vector routing maximum (the existing ISA lacks
`vfirst.m`).

The mapping deliberately serializes phases and drains DMA before changing its
source backend. It uses remote NoC reads/writes, with no collectives or overlap
optimization yet. Expert imbalance and small decode batches therefore leave
compute units idle; performance is a baseline for subsequent mapping work.

## Validation status

With the default architecture, layer 1, and seed 19, `decode_b1_kv1024` passed
all 36,296 numerical checks in 10,439,993 cycles (10.440 ms at 1 GHz), including
DMA and barriers. Generation and compilation of `prefill_s64` were also checked.
The remaining ten presets have not completed full simulator validation.
