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
and FP16 attention probabilities. Final device-side numerical validation is
disabled by default. With `VALIDATE=1`, results are checked with `atol=0.003`,
`rtol=0.02`, and exact expert indices. The checks cover each intermediate tensor,
each expert contribution, every output element, route weights, and appended K/V.
Outputs start as NaNs to expose missing writes. Host generation builds the NumPy
reference to select experts for preload and supply the optional validation data.

## Running

Use the SDK's RISC-V toolchain and an already built `soft_hier_old` simulator with
DRAMSys storage enabled. The host needs Python 3.9+ and NumPy. Set the usual GVSoC
and DRAMSys environment as for `apps_dram3d`; the makefiles only build software.

From this directory:

```sh
make plan CASE=decode_b16_kv4096    # Show addresses/capacity without generating weights
make -C decode run CASE=decode_b1_kv1024
make -C decode run CASE=decode_b1_kv1024 VALIDATE=1  # Enable final numerical checks
make -C decode suite               # B=1,8,16 × past KV=1024,2048,4096
make -C prefill run CASE=prefill_s64
make -C prefill suite              # B=1 × prompt length=64,128
make suite                         # All eleven full-size cases
```

All eleven presets use the full model dimensions. Decode case names follow
`decode_b{1,8,16}_kv{1024,2048,4096}`; prefill cases are `prefill_s64` and
`prefill_s128`. Defaults are `CASE=decode_b1_kv1024`, `SEED=19`, `LAYER=1`,
`OPT_STEP=25`, `VALIDATE=0`, and `TIMEOUT=3600` seconds. Override `ARCH` or `GVSOC` to select
the configuration or simulator path. `OUT` defaults to `build/<case>-l<layer>-s<seed>`.

After generation, `make compile` and `make simulate` reuse the preload. Keep
the same `CASE`, `LAYER`, and `SEED`, or pass the generated directory as `OUT`.
`VALIDATE` is a compile-time option: use `make compile VALIDATE=1` to enable
checking, or `make compile VALIDATE=0` to disable it. `make simulate` uses the
mode already compiled into `layer.elf`. External runners that require
`LLM_DRAM3D_PASS` must build with `VALIDATE=1`; export this variable when a runner
invokes Make internally.

The generated `manifest.json` gives tensor offsets, dimensions, capacity,
preloaded experts, and expected validation counts. `simulation.log` contains
diagnostics. A default run ends with `LLM_DRAM3D_DONE` and `checked=0`;
`results.json` records `status="DONE"` and `verified_elements=0`. With
`VALIDATE=1`, a successful run reports `LLM_DRAM3D_PASS`, and the runner requires
the full expected check count before recording `status="PASS"`. Both modes
require complete total and per-stage cycles at 1 GHz. Timing includes layer
DMA, computation, and synchronization; preload and optional numerical checking
are excluded. Simulator exit status alone is not accepted as completion.
The legacy core exposes a 32-bit `cycle` CSR. Each stage uses a wrapping unsigned
difference (each stage must take fewer than 2³² cycles), and the total is accumulated
in 64 bits. No compressed-instruction libm/libgcc routines are called.

## Recovering optimization steps

`OPT_STEP=N` selects a cumulative decode implementation at compile time; `0`
retains the original schedule and `25` is the default. Prefill keeps its original kernels. These steps
retain K=128 accumulation and FP16 spills, the reference, and validation tolerances.
Steps 13–17 change the B1 expert schedule; step 22 extends channel striping to
B8/B16. Regenerate older preloads once with `make generate` before using B1
steps 16+ or B8/B16 steps 22+.

```sh
make generate CASE=decode_b1_kv1024 OUT=build/compare
make benchmark OUT=build/compare OPT_STEPS="$(seq 0 26)"
make compile OUT=build/compare OPT_STEP=7 VALIDATE=1
make simulate OUT=build/compare
```

`benchmark` always enables validation and reuses one generated preload. Each
`OUT/steps/NN/` retains `source/`, `layer.elf`, `compile.log`, `simulation.log`,
`build.json` (commands and input/source SHA-256), and `results.json` (step, total
and eleven stage cycle counts, binary SHA-256, and host duration). It refuses to
overwrite a step directory. Use another `OUT` for a new comparison, or select
only unmeasured numbers in `OPT_STEPS`. To rebuild a saved source snapshot:

```sh
make -f build/compare/steps/07/source/Makefile compile \
  SDK_DIR="$PWD/../.." OUT="$PWD/build/compare" \
  ELF="$PWD/build/compare/steps/07/rebuilt.elf" OPT_STEP=7 VALIDATE=1
make simulate OUT="$PWD/build/compare" ELF="$PWD/build/compare/steps/07/rebuilt.elf"
```

The snapshot preserves the exact measured source even after later edits. Keep
the generated inputs and the same SDK runtime, toolchain, architecture, and
simulator when reproducing a historical cycle count.

B1 cycles below use `decode_b1_kv1024`, layer 1, seed 19, and `VALIDATE=1`.
Every row passed all 36,296 checks. Trials that regressed remain selectable;
step 25 is the fastest measured configuration and remains the default.

| Step | B1 cycles | Direction | Reason |
|---|---:|---|---|
| [00](sw/main.c#L15) | 10,439,995 | Original schedule | Establish the baseline. |
| [01](sw/main.c#L109) | 10,435,262 | Skip idle clusters | Avoid unused norm weights and projection inputs. |
| [02](sw/main.c#L124) | 10,435,267 | Clear padding only | Input DMA overwrites the live region. |
| [03](sw/main.c#L141) | 10,358,177 | Direct single-row input | Remove the repeated A-to-X copy when M=1. |
| [04](sw/main.c#L128) | 10,356,214 | Reuse RedMule configuration | Projection tile dimensions stay constant. |
| [05](sw/vector.h#L54) | 10,289,930 | Fuse half-precision addition | Keep the FP32 bias/residual sum in registers. |
| [06](sw/main.c#L173) | 10,298,844 | Tile-local SwiGLU | Consume gate/up pairs before writing them to DRAM. |
| [07](sw/main.c#L406) | 10,295,788 | Local expert activation | Retain single-worker activations in TCDM. |
| [08](sw/main.c#L113) | 10,296,896 | Pack input once | Gather directly into RedMule's K-tile order. |
| [09](sw/main.c#L151) | 6,723,340 | Double-buffer weights | Load the next K tile while RedMule computes. |
| [10](sw/main.c#L156) | 6,722,488 | Prefetch bias | Overlap the bias fetch with the last multiply. |
| [11](sw/main.c#L249) | 6,722,481 | Trim attention clears | Only incomplete cache blocks need zero padding. |
| [12](sw/main.c#L275) | 6,719,478 | Batch softmax rescaling | Evaluate the eight head-rescale exponentials together. |
| [13](sw/main.c#L357) | 6,007,748 | Separate B1 expert workers | Avoid serial execution when expert owners collide. |
| [14](sw/main.c#L359) | 5,971,369 | Two workers per expert | Split output-column tiles across two clusters. |
| [15](sw/main.c#L359) | 6,033,059 | Four workers per expert | Use all sixteen clusters for the four selected experts. |
| [16](scripts/generate.py#L90) | 2,225,356 | Stripe expert weights | Spread output tiles over all DRAM channels. |
| [17](sw/main.c#L358) | 1,289,063 | Match workers to channels | Process selected experts in turn with sixteen local readers. |
| [18](sw/main.c#L158) | 1,277,706 | Continue weight prefetch | Carry the pipeline across output-tile boundaries. |
| [19](sw/vector.h#L153) | 1,269,930 | Fuse RMSNorm passes | Remove widened input, weight, and square temporaries. |
| [20](sw/vector.h#L204) | 1,262,551 | Fuse RoPE loads | Widen input and table values directly in registers. |
| [21](sw/vector.h#L83) | 1,255,064 | Fuse expert combination | Accumulate weighted half rows without a widened temporary. |
| [22](sw/main.c#L384) | 1,255,064 | Stripe batched experts | Reuse all sixteen channels for each routed B8/B16 expert. |
| [23](sw/main.c#L112) | 1,264,387 | Specialize single-row gathering | Use one contiguous input transfer when M=1. |
| [24](sw/vector.h#L251) | 1,260,147 | Fuse bias and SwiGLU | Keep the FP16-rounded bias result in registers. |
| [25](sw/main.c#L374) | 1,231,649 | Share activation through TCDM | Remove the shared activation's DRAM round trip. |
| [26](sw/main.c#L112) | 1,235,521 | Retry tiled B1 gathers | Test the earlier gather with shared TCDM activation; slower than step 25. |

## Mapping and execution

| Object | Placement |
|---|---|
| Channel address | `0x10000000000 + cid * 0xc0000000 + offset` |
| Dense projection weights | 64-column tiles, tile `j` on channel `j % 16` |
| Expert weights, B1 steps 0–15 / B8/B16 steps 0–21 | Expert `e` on channel `e % 16`; eight experts per channel |
| Expert weights, B1 steps 16+ / B8/B16 steps 22+ | Additional copy with output tile `j` on channel `j % 16` |
| Token rows, layer I/O, intermediates | Flattened token `t` on channel `t % 16` |
| KV cache | Entire `(batch, KV-head)` on channel `(batch*8 + head) % 16` |
| Expert scratch | Step 7+ uses local activation for single workers; teams use leader DRAM until step 25, then leader TCDM |
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
For decode, both expert layouts are preloaded so every step uses the same data;
the extra copy does not move the original tensors. At KV1024 the reserved space
for B1 is 845,907,456 bytes/channel and 299,264 bytes of TCDM. Later steps read only the
striped copy, so retaining the original does not double timed weight traffic.

RedMule computes dense/expert projections and attention QK/PV products. Tiling is
M≤16, N=64, K=128 for projections, with explicit zero padding. Attention streams
128 cache positions at a time, uses the transpose engine for K, and maintains an
FP32 online softmax denominator and output accumulator, including each head's
sink logit in the denominator only.

Spatz vector code handles RMSNorm, split-half RoPE, softmax reductions and
exponentials, router maximum reductions, SwiGLU, biases, and residual/expert
combination. It follows the SDK's `implementation/sw/RMSNorm`, `Activation`, and
`FlatAttention` instruction patterns, including custom `vfexp.vv`. The schedule
uses core 0's Spatz and RedMule in each active cluster. Other cores join
barriers. Scalar code handles DMA/control, one-value softmax state updates, and
the index scan after each vector routing maximum (the existing ISA lacks
`vfirst.m`).

DMA drains before changing source backends. Weight DMA overlaps RedMule only
through separate buffers. Expert teams publish completion epochs in synchronization
memory after DMA finishes; shared activation is consumed only after all producers
arrive. Other cores continue to participate in the existing stage barriers.

## Validation status

Measurements use the SDK architecture at 1 GHz, GVSoC fast cores, direct preload,
and DRAMSys storage with power accounting disabled. Baseline is step 0; default
is step 25. All runs below used `VALIDATE=1`, with checks outside the timed layer.
At KV1024, B1/B8/B16 speedups are **8.48× / 3.94× / 2.66×**, respectively.

| Case | Layer / seed | Baseline cycles | Default cycles | Checks |
|---|---|---:|---:|---:|
| `decode_b1_kv1024` | 1 / 19 | 10,439,995 | 1,231,649 | 36,296 |
| `decode_b8_kv1024` | 1 / 19 | 25,592,617 | 6,489,178 | 290,368 |
| `decode_b16_kv1024` | 1 / 19 | 31,076,012 | 11,665,375 | 580,736 |
| `decode_b1_kv4096` | 1 / 19 | — | 1,452,040 | 36,296 |
| `decode_b1_kv4096` | 0 / 7 | — | 1,156,409 | 36,296 |

A dash means no matching baseline was recorded in this run set. The default
`VALIDATE=0` B1 run also completed: `DONE`, zero final checks, 1,229,765 cycles.
Other decode presets and prefill were not fully simulated in this sweep.
All eleven generated layouts fit channel/TCDM capacity, and the existing 18
`experiments/llm_dram3d` compatibility tests pass.

Local artifacts are retained under `build/optimization/`: `series_b1/steps/`
contains all 27 B1 configurations, `b1_steps.csv` includes all eleven stage
measurements, and `environment.json` records the compiler, runtime, and simulator
provenance. The B8/B16 original-layout runs are in `decode_b{8,16}_kv1024/steps/`;
striped-layout runs are in `series_b{8,16}/steps/`. Original tensor addresses and
bytes were checked against the striped preloads. Rebuilding saved source snapshots
also reproduced their executable code exactly.
