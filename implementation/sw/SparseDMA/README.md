# Sparse DMA on SoftHier

Run the three gather cases on cluster 0 through the NoC to west HBM node 0:

| Case | DMA commands |
| --- | --- |
| `core-loop` | One SDK DMA call per row |
| `inlined-loop` | Inline DMA instructions per row |
| `hw-gather` | One gather descriptor with packed 16-bit indices |

The workload selects 128 rows × 256 bytes from a 2048 × 128 FP16 matrix,
using seed 42. Source addresses remain `0xc0100000` (matrix) and `0xc0180000`
(indices).

`config/arch/sparse_dma.py` uses 4 × 4 clusters, five cores per cluster,
384 KiB TCDM, 64 iDMA transactions, 256 burst slots and a 1024-bit NoC.
It enables gather and selects `HBM2E-3600.json`. The test environment uses the
standalone HBM2E configuration and DRAMSys library under `gvsoc/build/sparse_dma/`.
Only cluster 0/core 0 runs; the other cores wait in WFI.

From `softhier-sdk/implementation`:

```bash
make sparse-runv       # Generate data, build, simulate all cases and report
make sparse-report     # Regenerate the report from existing runs
```

The wrapper uses the workspace GVSoC environment and SDK-local RISC-V toolchain.
For individual steps, use `make sparse-cfg`, `make sparse-hw`, `make sparse-sw`,
and `bash scripts/sparse_dma/step.sh run <case>`.

Checks cover bit-exact output, guards, descriptor counts, the NoC/HBM route and
HBM burst timing. Cycles include DMA setup, issue and completion wait; preload,
verification and printing are excluded.

Results, plots, binaries and traces are under `build/sparse_dma/`. These are
SoftHier sanity measurements, without an RTL cycle-accuracy claim.
