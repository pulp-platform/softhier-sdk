# Sparse attention access

Each cluster gathers selected tokens from its own independent context into local
TCDM. The benchmark compares an SDK DMA call per token, an inlined DMA loop and
hardware gather. It models the access phase only; no attention arithmetic runs.

## Run

From `softhier-sdk/implementation` in the initialized GVSoC workspace:

```bash
make sparse_attn_access-run
```

This builds the SoftHier model and three programs, prepares their preload ELFs,
runs each case, checks results and generates the report. It uses the same workspace
environment and HBM2E DRAMSys library as `sparse_dma`.

Outputs are in `build/sparse_attn_access/`:

- `REPORT.md`, `results.csv`, `results.json`: runtime, HBM utilization and audits.
- `runtime_hbm.*`, `hbm_nodes.*`: runtime and bandwidth plots (PNG/SVG/PDF).
- `noc_core_loop.*`, `noc_inlined_loop.*`, `noc_hw_gather.*`: NoC traffic plots.
- `clusters.csv`: per-cluster cycle counts and global execution windows.
- Each case directory: program, preload, simulation log and DRAMSys databases.

Individual targets are `sparse_attn_access-cfg`, `-hw`, `-sw`, `-pld`, `-pre`
and `-report`. For one prepared case:

```bash
bash scripts/sparse_attn_access/step.sh run hw-gather
make sparse_attn_access-report
```

Saved runs are protected against accidental overwrite. Choose another output
directory for a new run:

```bash
make sparse_attn_access-run SPARSE_ATTN_OUTPUT="$PWD/build/sparse_attn_access_repeat"
```

## Configuration

Edit [`config/kernels/sparse_attn_access.py`](../../config/kernels/sparse_attn_access.py):

| Parameter | Default |
| --- | --- |
| `token_dtype` | `"fp16"` (`"fp32"` also supported) |
| `token_dim` | 128 |
| `selected_tokens` (N) | 1024 |
| `context_tokens` (M) | 131072 |
| `selection_seed` | 42 |
| `sorted_indices` | `True` |

Selections contain N unique tokens, using a reproducible NumPy generator seeded
by `(selection_seed, cluster_id)`. Sorting is by logical token index. All three
cases share identical selections, data and output order. Indices are in L1 before
the measured region. FP16/FP32 values are checked bit for bit without arithmetic.

The selected output and both logical and mapped index lists must fit TCDM.
Token size must be a power of two between 32 and 4096 bytes. Invalid capacity
or address configurations are rejected before simulation.

The architecture is [`config/arch/sparse_attn_access.py`](../../config/arch/sparse_attn_access.py),
inherited from `sparse_dma`: 4 × 4 clusters, five cores per cluster, 384 KiB TCDM,
64 DMA transactions, 256 burst slots, a 1024-bit NoC and four west HBM2E nodes.
For a larger square mesh, change only these entries:

```python
self.num_cluster_x = 8
self.num_cluster_y = 8
self.hbm_chan_placement = [8, 0, 0, 0]
```

Use 16 in all three positions for 16 × 16. Alternate configuration files can be
passed with `SPARSE_ATTN_ARCH_FILE` and `SPARSE_ATTN_KERNEL_FILE`. Hardware builds
share the installed GVSoC model, so build and run different architectures serially.
Only the default 4 × 4 case is included in the initial measured report.

### HBM on all four edges

`config/arch/sparse_attn_access_hbm16.py` changes only HBM placement to
`[4,4,4,4]`, giving 16 nodes around the same 4 × 4 cluster mesh:

```bash
make sparse_attn_access-run \
  SPARSE_ATTN_ARCH_FILE="$PWD/config/arch/sparse_attn_access_hbm16.py" \
  SPARSE_ATTN_OUTPUT="$PWD/build/sparse_attn_access_hbm16" \
  SPARSE_ATTN_BASELINE="$PWD/build/sparse_attn_access"
```

The optional baseline produces comparison plots and `comparison.csv`, after
checking that selections, payloads and all architecture parameters except HBM
placement match. Replace `-run` with `-report` to regenerate saved reports.

Node IDs follow the chip address map: W0–W3 = 0–3, N0–N3 = 4–7,
E0–E3 = 8–11, S0–S3 = 12–15. Token `t` maps to node `t % 16`.
Each node holds 32 MiB of context space; each case still reads 4 MiB in total.
The flow supports west-only placement or one HBM channel per node on all four edges.

For 128 selected tokens out of a 1,024-token context on the same 16-node architecture:

```bash
make sparse_attn_access-run \
  SPARSE_ATTN_ARCH_FILE="$PWD/config/arch/sparse_attn_access_hbm16.py" \
  SPARSE_ATTN_KERNEL_FILE="$PWD/config/kernels/sparse_attn_access_m1024_n128.py" \
  SPARSE_ATTN_OUTPUT="$PWD/build/sparse_attn_access_hbm16_m1024_n128"
```

This reads 32 KiB per cluster and 512 KiB total, with one HW-gather descriptor
per cluster. FP16 × 128, seed 42 and sorted indices remain the same.

## Context layout

For H enabled HBM nodes, user/cluster c and logical token t:

```text
node = t % H
tokens_per_user_on_node = ceil((M - node) / H)
local_offset = 1 MiB + (c * tokens_per_user_on_node + t // H) * token_bytes
physical_address = HBM_base + node * HBM_node_window + local_offset
```

Contexts are adjacent within every node, with no gaps or overlap between users.
The first 1 MiB is reserved for the SDK runtime. Defaults give 32 MiB per context,
512 MiB of context address space, 128 MiB on each node and 4 MiB read per case.
Only selected token values are initialized; unselected addresses are never read.

Hardware gather uses the existing 32-bit index mode, with mapped index
`physical_address / token_bytes`, source base zero and token-sized stride. This
preserves logical order across nodes, including addresses above 4 GiB, without
changing the DMA model. The preload uses ELF64 physical addresses; the software
itself remains RV32. Larger meshes use multiple preload ELFs to stay below the
ELF segment-count limit without filling unused context gaps.

## Measurement

One core per cluster runs the access loop. All cores use the SDK barriers for
synchronization. Every case uses 128-token batches at the default queue sizes,
waiting for each batch before issuing the next. This keeps the original private
iDMA backend below queue exhaustion. HW gather issues eight descriptors per
cluster; the software loops issue 1024.

Total runtime is the span from the earliest cluster start to the last completion,
measured by existing cluster annotation registers. Preload, index preparation,
output initialization and verification are excluded. Per-cluster `mcycle` values
provide a separate timing check. Printing is serialized outside the timed region.

HBM utilization is verified read bytes divided by this runtime and the aggregate
peak from the recorded DRAMSys specification. HBM2E's recorded 556 ps clock gives
57.554 GB/s per node, 230.216 GB/s across four nodes or 920.863 GB/s across 16.
The analysis checks burst
spacing and overlap separately for both pseudo-channel buses.

With `SOFTHIER_DATA_NOC=floonoc_v2`, the plots include separate request and
read-data return networks. Select this backend for both hardware build and run;
see the [implementation README](../../README.md#floonoc-v2).

For the default legacy backend, NoC arrows show measured read-request hops per microsecond; heatmaps show payload
bytes requested by each cluster from each HBM node. The legacy model returns read
data through callbacks, without separately routed response links. The plots
therefore describe request traffic, not measured return-link utilization.

The report requires bit-exact outputs, index hashes, guard regions, descriptor
counts and exact DMA/NoC/HBM read addresses and volumes to pass. Full-chip results
use SDK fast cores and do not establish RTL cycle accuracy for this new workload.

The shared ELF loader must resume an asynchronous completion on a following
clock edge (`core/models/utils/loader/loader.cpp`). Otherwise the many-segment
preload can trigger a zero-time SystemC stall. This loader fix affects setup only.
