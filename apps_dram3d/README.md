# FP16 GEMM with memory on logic

This application uses one independent DRAMSys channel above each cluster.
`config/arch.py` copies `implementation/config/arch/arch.py`: 4×4 clusters,
5 cores per cluster, four Spatz cores, 384 KiB TCDM, a 32×16 RedMule engine,
and a 1024-bit NoC at the board's 1 GHz clock. It disables all side HBM and
enables:

```python
self.dram3d_enable = 1
self.dram3d_type = 'hbm2-example.json'
self.dram3d_addr_base = 0x10000000000
self.dram3d_node_space = 0xc0000000
```

Python attribute names cannot start with a digit, hence `dram3d_*`. Channel
`cid = y * 4 + x` starts at `dram3d_addr_base + cid * dram3d_node_space`.
There is no HBM aliasing or channel interleaving. The 3 GiB aperture spacing
does not enlarge the HBM2 memspec's 2 GiB capacity. DMA addresses remain
64-bit throughout; RedMule operates on local TCDM buffers.

## Build and run

Use this SDK checkout as `soft_hier_sdk/` inside the GVSoC repository. Build
GVSoC with the updated `pulp.chips.soft_hier_old.flex_cluster` target and the
repository's matching patched DRAMSys/SystemC libraries. The DRAMSys config
must have storage enabled (`StoreMode: Store`). The bundled prebuilt library
may need rebuilding as described in the hardware repository's SoftHier README.
Python needs NumPy; the SDK's RISC-V compiler must be installed.

From the GVSoC root, with the usual simulator environment set:

```sh
make -C soft_hier_sdk/apps_dram3d hw
make -C soft_hier_sdk/apps_dram3d run CASE=big1024
make -C soft_hier_sdk/apps_dram3d suite
```

`hw` selects the preset with `SOFTHIER_ARCH_FILE`; it does not overwrite the
hardware's default architecture file. Architecture headers, a local-data linker
script, application ELF, and preload ELF are generated under this application's
ignored `build/` directory. Existing runtime headers are left untouched.

| Case | M×N×K | Default dataflow | RedMule tile M×N×K | TCDM including 4 KiB reserved |
| --- | --- | --- | --- | ---: |
| `big1024` | 1024×1024×1024 | SUMMA | 128×128×256 | 324 KiB |
| `big2048` | 1024×2048×1024 | SUMMA | 128×128×256 | 324 KiB |
| `flat16` | 16×1024×1024 | Local N partition | 16×64×1024 | 166 KiB |
| `flat8` | 8×2048×1024 | Local N partition | 8×128×1024 | 278 KiB |

Useful overrides are `PYTHON`, `CC`, `GVSOC`, `ARCH`, `OUT`, `SEED` (19),
`TRACE_REDMULE` (0), and `TIMEOUT` (600 host seconds). These presets require a 4×4 grid and at least
two cores per cluster. Change `SEED` to generate another complete input set.

To compare the large kernels against a layout that reads every panel locally:

```sh
make -C soft_hier_sdk/apps_dram3d run CASE=big1024 DATAFLOW=local
make -C soft_hier_sdk/apps_dram3d run CASE=big2048 DATAFLOW=local
```

`DATAFLOW=auto` selects SUMMA for large cases and local N partitioning for flat
cases. `DATAFLOW=summa` explicitly selects SUMMA for the large cases.

Each run writes `simulation.log`, `manifest.json`, and `results.json` under
`build/<case>-<dataflow>/`. A nonzero simulator status, timeout, missing PASS
marker, or any output mismatch fails the command. `results.json` records
cycles, useful MACs/cycle, fraction of the 16-engine peak, and effective DRAM
payload GB/s. The payload metric counts requested kernel bytes, excluding
preload and verification; it is not a DRAM bus command counter.

Set `TRACE_REDMULE=1` to enable the model's per-call timing trace and write
`redmule_timing.json`. `results.json` then also records `redmule_active_fraction`
(time inside RedMule calls divided by kernel time) and `redmule_call_efficiency`
(useful MACs divided by the peak capacity during those calls). Their product
is `system_peak_fraction`. Call time includes the engine's internal TCDM
preload and drain, so activity alone does not imply full MAC throughput.

## Layout and scheduling

For SUMMA, let `i`, `j`, and `s` be global M-, N-, and K-tile indices.
`A[i,s]` resides in channel `(s % 4, i % 4)` and `B[s,j]` in channel
`(j % 4, s % 4)`. Cluster `(j % 4, i % 4)` owns `C[i,j]`. A panels are
broadcast along a mesh row; B panels along a column. Roots rotate each K step,
spreading reads across all sixteen stacked channels. Each channel stores its
panels contiguously, so all panel transfers are one-dimensional bursts.

Two A and B buffers allow the DM core to fetch and broadcast the next panels
while core 0 runs RedMule on the current pair. Prefetching continues across
output-tile boundaries. Two C buffers allow the previous result to be written
while RedMule starts the next output tile; each C tile stays in TCDM across K
steps and is written once. DMA clears the next C buffer from `zomem(0)` during
the preceding tile's last K step. RedMule dimensions are configured once.

Global barriers protect buffer reuse and collective completion. Broadcasts with
different masks are explicitly drained before changing masks, matching the
legacy DMA backend's collective metadata lifetime. C stores are also drained
before switching the DMA source backend from TCDM to DRAM. Both the store and
subsequent prefetch overlap RedMule execution on the other C buffer. The SUMMA
root that reads both step-1 panels defers its previous-C store by one step so
that the store does not compete with those two panel reads on its channel.

The local alternative replicates A across a cluster row and B down a column.
It uses 256×128×128 tiles with the same overlap schedule and 324 KiB TCDM
footprint, while each cluster reads its own stacked channel. The taller output
tile reduces repeated B-panel reads. It exposes all sixteen DRAM channels
concurrently and removes the panel broadcasts, at the cost of four times the
stored input data and three times the kernel input-read traffic of the SUMMA
preset. Output tiles retain block-cyclic ownership over the 4×4 grid.

For flat GEMM, cluster `cid` owns columns
`[cid * (N/16), (cid+1) * (N/16))`. Its full K×(N/16) weight slice and a copy
of the small M×K activation matrix fit in TCDM. It loads both once, performs
one RedMule call, and stores its output slice to its own channel. All sixteen
clusters participate. The small M dimension limits RedMule row utilization;
the layout primarily exploits aggregate stacked-channel bandwidth.

Within each channel, packed A, B, C, and the validation reference occupy
successive 128-byte-aligned regions. `gemm.h` contains their offsets, and
`preload.elf` uses ELF64 physical addresses. The program's normal data is in
local TCDM; the legacy runtime linker script's side-HBM `.data` placement is
not used.

## Correctness and measurements

The generator produces independent, varied inputs from `{-1/8, 0, 1/8}` and
computes `A @ B` with NumPy in FP32. For these shapes, every product and partial
sum is exactly representable in FP16, so every output is checked exactly
(signed zero is equivalent). This also avoids comparing different reduction
rounding orders. For arbitrary FP16 inputs, the model rounds accumulators at
its RedMule tile boundaries; a full FP32 reference may need a tolerance.

C is preloaded with NaNs, then explicitly zeroed by the kernel. Verification
reads every stored output back from DRAM and compares it with the independently
generated reference. A single core reports per-cluster errors to avoid
interleaved diagnostics. ELF loading and validation are outside the timed region;
input transfers, broadcasts, barriers, computation, and output stores are inside.

Measured on the configuration above, with direct preload and the fast core:

| Shape | Dataflow | Cycles | Useful MAC utilization | Checked outputs |
| --- | --- | ---: | ---: | ---: |
| 1024×1024×1024 | SUMMA | 145,785 | 89.9% | 1,048,576 |
| 1024×1024×1024 | Local | 145,186 | 90.3% | 1,048,576 |
| 1024×2048×1024 | SUMMA | 281,549 | 93.1% | 2,097,152 |
| 1024×2048×1024 | Local | 282,092 | 92.9% | 2,097,152 |
| 16×1024×1024 | Local | 13,739 | 14.9% | 16,384 |
| 8×2048×1024 | Local | 24,333 | 8.4% | 16,384 |

All rows passed full validation. The two dataflows differ by less than 0.5%
in latency for these large cases. SUMMA uses one third of the local preset's
input-read traffic and one quarter of its stored input data. These figures
describe this simulator configuration, not a physical F2F implementation;
the interface has no additional PHY latency.

The initial 128×128×256 SUMMA kernel took 176,838 cycles (74.1% of peak).
A trace of cluster 0 separates that time as follows:

| Part of the original kernel | Cycles |
| --- | ---: |
| Useful MACs at the engine's peak rate | 131,072 |
| Internal RedMule preload/drain overhead | 2,048 |
| Three output-tile boundary gaps | 31,471 |
| Gaps between K-panel calls within an output tile | 1,653 |
| Initial fill and final drain, including surrounding control | 10,594 |
| Total | 176,838 |

Each of cluster 0's sixteen calls took 8,320 cycles for 8,192 ideal compute
cycles: 98.5% efficiency inside a call. The missing throughput came primarily
from stopping the prefetch pipeline at output-tile boundaries. Continuing it
and double-buffering C removed those long stalls. The updated SUMMA run has
91.5% engine activity and 98.3% efficiency inside calls, giving 89.9% useful
MAC utilization over the complete timed kernel. Cluster 0's interval from its
first engine launch through its last engine store spans 135,178 cycles, or
97.0% useful MAC utilization; initial fill, final DMA drain, and surrounding
control account for the other 10,607 cycles. Those costs remain inside the
full-kernel measurement. The larger 2048-column case amortizes them and reaches
93.1%.
