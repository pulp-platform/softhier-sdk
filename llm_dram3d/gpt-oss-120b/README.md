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
make -C decode run CASE=decode_b1_kv1024 BASELINE=1  # Original schedule
make -C decode suite               # B=1,8,16 × past KV=1024,2048,4096
make -C prefill run CASE=prefill_s64
make -C prefill suite              # B=1 × prompt length=64,128
make suite                         # All eleven full-size cases
make test                          # Host layout/packing/report regressions
```

All eleven presets use the full model dimensions. Decode case names follow
`decode_b{1,8,16}_kv{1024,2048,4096}`; prefill cases are `prefill_s64` and
`prefill_s128`. Defaults are `CASE=decode_b1_kv1024`, `SEED=19`, `LAYER=1`, and
`TIMEOUT=3600` seconds. Override `ARCH` or `GVSOC` to select the configuration or
simulator path. `OUT` defaults to `build/<case>-l<layer>-s<seed>`.
Decode uses the optimized schedule by default. `BASELINE=1` selects the original
schedule and adds `-baseline` to the default output directory. Prefill always
uses the original schedule.

After generation, `make compile` and `make simulate` reuse the preload. Keep
the same `CASE`, `LAYER`, and `SEED`, or pass the generated directory as `OUT`.
Regenerate an older decode preload with `make generate compile` to enable the
new expert layout. `compile` alone preserves the schedule in the generated
header, including compatibility with original preloads. Never mix a binary and
preload from different schedules.

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
| Expert weights, optimized decode | Output tile `j` of expert `e` on channel `(j + e) % 16` |
| Expert weights, baseline/prefill | Expert `e` on channel `e % 16`; eight experts per channel |
| Token rows, layer I/O, intermediates | Flattened token `t` on channel `t % 16` |
| KV cache | Entire `(batch, KV-head)` on channel `(batch*8 + head) % 16` |
| Expert activations, optimized decode | Four slots per token, in the token owner's channel |
| Expert scratch, baseline/prefill | Reused up/gate and activation buffers in the expert owner's channel |
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
Increase `TIMEOUT` for long full-size or baseline simulations if necessary;
`TIMEOUT=14400` allows four hours of host simulation time without changing the
simulated cycle measurement.

RedMule computes dense/expert projections and attention QK/PV products. Tiling is
M≤16, N=64, K=128 for projections, with explicit zero padding. Attention streams
128 cache positions at a time, uses the transpose engine for K, and maintains an
FP32 online softmax denominator and output accumulator, including each head's
sink logit in the denominator only.

Spatz vector code handles RMSNorm, split-half RoPE, softmax reductions and
exponentials, router maximum reductions, SwiGLU, biases, and residual/expert
combination. It follows the SDK's `implementation/sw/RMSNorm`, `Activation`, and
`FlatAttention` instruction patterns, including custom `vfexp.vv`. The schedules
use core 0's Spatz and RedMule in each active cluster. Other cores join
barriers. Scalar code handles DMA/control, one-value softmax state updates, and
the index scan after each vector routing maximum (the existing ISA lacks
`vfirst.m`).

## Decode optimization

The original schedule places an entire expert on one channel. A small decode
batch activates few clusters and can select several experts on the same cluster.
The optimized schedule stripes each expert across all 16 channels, rotating the
stripe by expert id to balance partial stripes. It reserves all 128 experts and
uses the device's computed routes; scheduling never relies on reference routes.
Weights are distributed without replication.

The gate/up sweep applies bias and SwiGLU directly to each completed 64-column
tile, then writes 32 activation elements to the appropriate token/top-k slot.
A single global barrier makes these activations available to the down sweep.
This removes the full gate/up DRAM write/read and the per-expert synchronization
that a sequential distributed implementation would require. All intermediate
values already covered by the numerical checker remain checked.

**Batch weight reuse is preserved.** Dense projections process up to 16 token
rows with one weight load. Both expert sweeps gather all tokens selecting the
same expert into an M tile. Across the supported decode batches (B≤16), every
selected expert's weight tile is therefore read once, regardless of how many
tokens selected it. Expert outputs and route probabilities remain distinct for
each token and top-k slot.

Inputs are gathered once in `[K tile][token row][128]` order, eliminating the
per-GEMM TCDM input repack. Each input row's gather is drained before queuing
the next row: queuing an entire batch can stall the legacy DMA under
backpressure, observed at the K=4096 output projection. Two weight buffers
hold eight K tiles each for B=1/8 or four for B=16. DMA of the next panel
overlaps RedMule computation, including
across output-tile boundaries. Bias addition uses a single widening vector
addition and one FP16 conversion. The K=128 accumulator spills, FP32 nonlinear
operations, reference implementation, and validation tolerances are unchanged.
DMA is drained before switching source backends, and a vector-store interlock
precedes DMA reads of vector results.

TCDM use is 356,736 / 366,976 / 301,696 bytes for B=1 / 8 / 16, within the existing
393,216-byte capacity. The largest decode preset reserves 446,667,392 bytes per
DRAM channel. `make plan` reports the selected layout without generating weights.
No hardware configuration or simulator changes are required.

## Validation status

Measured with the unchanged default architecture, layer 1, and seed 19.
Both schedules use the same weights, inputs, NumPy reference, precision, and
tolerances. Timings include DMA, computation, and barriers.

| Case | Baseline cycles | Optimized cycles | Speedup | Numerical checks |
|---|---:|---:|---:|---:|
| `decode_b1_kv1024` | 10,439,993 | 1,241,073 | 8.41× | 36,296 |
| `decode_b8_kv1024` | 25,591,547 | 6,037,009 | 4.24× | 290,368 |
| `decode_b16_kv4096` | 32,864,187 | 12,370,639 | 2.66× | 580,736 |

For B=1, the expert stage decreases from 9,893,491 to 840,118 cycles (11.78×).
QKV decreases from 188,366 to 116,548 cycles; the output projection decreases
from 158,981 to 101,050 cycles. Total layer latency is 1.241 ms at 1 GHz,
an 88.1% reduction. A fresh `BASELINE=1` run reproduces the original baseline.

The optimized sliding-window case `decode_b1_kv4096 LAYER=0 SEED=7` also passes
all 36,296 checks in 1,160,448 cycles. Host regressions cover all preset capacity
plans, original-layout compatibility, rotated weight packing with K/N tails,
preloads with more than 2,048 ELF segments, and binary/preload schedule matching.
The unchanged prefill path compiles for `prefill_s64`; a full prefill simulation
has not been repeated for this decode optimization.
Full simulator coverage for this change is limited to the three table entries
and the sliding-window case (943,696 optimized numerical checks in total).
The remaining presets have capacity/layout validation.
