# GPT-OSS-120B: one decoder layer

This program decodes one new token per sequence through a complete full-attention
layer on `pulp.chips.soft_hier_old.flex_cluster`. A single `decode.elf` supports
batches 1 through 64. The benchmark uses batches 1, 8, and 64, each with 2,048
cached tokens and the current token at position 2,048. It uses reproducible
synthetic inputs, cache contents, and weights, with the full model dimensions.

The dimensions and equations follow OpenAI's [model configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)
and [PyTorch reference](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py),
checked on 2026-10-04. This is layer index 1, a full-attention layer. It has hidden
width 2,880, 64 query heads, 8 KV heads, head dimension 64, 128 experts,
top-4 routing, and expert intermediate width 2,880. The requested FP16 storage
replaces the checkpoint's native precision; checkpoint files are not required.

## Computation

`main.c` runs weighted RMSNorm, biased QKV projection, YaRN RoPE and cache append,
GQA with an attention sink per query head, biased output projection and residual,
a second RMSNorm, biased router projection, top-4 softmax, expert dispatch,
biased gate/up projection, clipped SwiGLU, biased down projection, and a weighted
expert sum with the final residual. RoPE uses split-half rotations, theta
150,000, factor 32, and the reference's fractional interpolation bounds.
Gate/up channels are interleaved; SwiGLU uses alpha 1.702 and limit 7.

All persistent tensors, weights, biases, norms, cache entries, and intermediate
results are FP16 in HBM. Normalization, softmax, and weighted sums use FP32 scalar
reductions. RedMule accumulates in FP32 within a launch and stores FP16 between
K tiles. This is a functional implementation: every selected expert matrix is
read and multiplied, with no host-side output injection or analytic timing shortcut.

Scheduling uses the SDK runtime and the DMA/RedMule patterns in
`implementation/sw/SummaGEMM` and `implementation/sw/FlatAttention`, along with
the attention configuration/preload code in `implementation/scripts/kernels`.
Each cluster's core 0 controls DMA, RedMule, and vector elementwise operations.
The other cores participate in synchronization. This is an initial baseline;
it does not use all four vector cores per cluster.

Linear layers distribute output-column tiles across the 16 clusters. Weights
are packed as `[N-tile, K-tile, 256, 128]`, padding partial dimensions with zeros.
Inputs and weights use double buffers in the 384 KiB TCDM. GQA distributes
`(sequence, KV-head)` tasks, sharing each K/V tile across its eight query heads.
K is transposed in HBM; V is row-major. Attention processes 128 cache positions
per tile and masks padding after the current token. Expert rows are grouped by
expert to reuse weights when multiple sequences choose the same expert.

## HBM layout and synthetic data

`config/arch.py` inherits `implementation/config/arch/arch.py`: 4×4 clusters,
5 cores per cluster, 128 TCDM banks, 32×16 RedMule, 1024-bit NoC, and four HBM
channels on each of the west and south edges, with XOR channel scrambling.
It increases the logical region per HBM node from 3 GiB to 4 GiB so that all
FP16 experts fit. Aliased nodes share storage. The evaluation's DRAM address
mapping covers 1 GiB per physical channel; compute resources and channel
bandwidth are unchanged.

The first 256 MiB on each populated edge is reserved for tensor workspaces;
weights start at offset `0x10000000`. Even experts occupy the west edge and odd
experts the south edge. `manifest.json` records every address, shape, byte count,
ELF offset, and parameter checksum. The full packed parameter payload is
6,903,704,960 bytes; maximum-batch tensor reservations total 295,998,020 bytes.

| Tensor | Shape at maximum batch | HBM address |
| --- | --- | --- |
| Input | `[64, 2880]` | `0xc0010080` |
| K cache, transposed | `[64, 8, 64, 2176]` | `0xc0164080` |
| V cache | `[64, 8, 2176, 64]` | `0xcc0010000` |
| Expert input | `[256, 2880]` | `0xc8af7300` |
| Expert gate/up | `[256, 5760]` | `0xcc8810000` |
| Expert activation | `[256, 2880]` | `0xcc8ae0000` |
| Expert output | `[256, 2880]` | `0xcc8c48000` |
| Layer output | `[64, 2880]` | `0xc8c5f300` |

`weights.elf` is common to all batches and contains all 128 experts.
`input-bN.elf` contains the runtime batch control, input, and initial KV cache.
The same sequence's input/cache prefix is identical across batches. Parameter
seeds are independent of batch size. Dense projection/expert weights are random;
the synthetic router is a fixed diagonal matrix with input markers that produce
separated, repeatable top-4 choices. Routing, dispatch, and expert computation
still execute on the simulator. Results depend on expert reuse and do not predict
the routing distribution of a trained checkpoint.

## Build and run in this repository

For a fresh checkout, [SETUP.md](SETUP.md) records the simulator revisions,
dependency versions, HBM configuration, and build commands used in evaluation.

From the GVSOC repository root, activate the locally prepared toolchain, Python,
SystemC, rebuilt DRAMSys library, and compatible DRAM configuration:

```sh
source build/direct_preload/env.sh
make -C soft_hier_sdk/LLM_decode/gpt-oss-120b regression
make -C soft_hier_sdk/LLM_decode/gpt-oss-120b smoke
make -C soft_hier_sdk/LLM_decode/gpt-oss-120b benchmark
```

`benchmark` generates data if needed, builds one executable, runs all three
batches, compares numerical outputs, and writes
`build/gpt_oss_decode/full/RESULTS.md`. Generation needs NumPy and approximately
7.3 GB of disk space. Simulation needs several GB of resident host memory and
can take substantial host time; preload consumes zero simulated cycles.
`BUILD_DIR`, `BATCHES`, and `PYTHON` are Makefile overrides. `make validate`
checks existing runs; `make data` explicitly regenerates all data.

The build script copies runtime headers into the build directory and generates
architecture definitions there. It does not modify shared SDK headers. The
program uses SDK startup, synchronization, DMA, RedMule, and dump APIs. It links
without the toolchain's compressed-instruction math library because this core
does not implement RVC. Scalar square root uses `fsqrt.s`; exponential uses
the SDK's custom vector instruction.

The simulator must include the associated fixes in `pulp/chips/soft_hier_old`:
Spatz LSU widths are converted from bits to bytes, FP16 RedMule conversion
preserves subnormals and rounds to nearest/even, and the bank arbiter accepts
subword DMA requests. Vector helpers wait for stores to complete before scalar
FP loads or independent DMA can consume results. The locally prepared build
includes these fixes and the default direct ELF preload implementation.
The non-Spatz core configuration also enables dependency checks for asynchronous
scalar loads; all eight preload regression cases pass after this correction.
The correction leaves the decoder smoke outputs and stage cycle counts unchanged.

The environment details and simulator rebuild dependencies are documented in
`build/direct_preload/RESULTS.md` in the GVSOC checkout. Both `--preload` and the
additional `hbm_preloader/binary` entry use the normal address mapping and
zero-cycle direct initialization. Cores are released at cycle 1.

## Validation and timing

The measured full-model results for batches 1, 8, and 64 are in
[RESULTS.md](RESULTS.md), with machine-readable records in
[results/measurements.json](results/measurements.json).

`tools/reference.py` independently computes the layer with NumPy from the exact
generated FP16 weights and inputs. It reproduces the documented FP16 tile
boundaries, requires exact expert IDs, and checks finite numerical output with
absolute tolerance 0.01 and relative tolerance 0.02. Reduced-dimension smoke
cases additionally dump every main intermediate tensor; batches 1 and 4 match
the reference exactly. These smoke results are correctness checks, not the
requested full-model timings.

`tests/vector.c` checks vector tails, short FP32 exponentials, completion ordering,
and strided 2-byte DMA with neighbor preservation. `tests/fp16_model.cpp` checks
every non-NaN FP16 round trip plus rounding, underflow, overflow, and NaN cases.

`LAYER_CYCLES` spans the complete layer, including DMA, compute, and barriers.
Fourteen stage counters sum to this total. At the configured 1 GHz clock,
one cycle is one nanosecond. Each stage uses a 32-bit cycle delta; the runner
cross-checks the total against the simulator's independent elapsed-time counter
to catch wraparound. The latter also includes a small constant reporting setup
overhead. Initialization and post-computation output dumps are excluded.

Each `batch-N` directory contains the simulator command, raw log, FP16 dump,
timing JSON, reference output, and validation JSON. The common executable's
SHA-256 is recorded in every result. See the generated `RESULTS.md` for measured
batch latency and the stage breakdown.

Independent batch runs can use separate simulator processes. A per-batch file
lock prevents simultaneous runs from overwriting each other's artifacts; a
duplicate request reuses the matching result produced while it was waiting.
