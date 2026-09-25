# Kernel for End-to-End Dataflow Implementation for Modern LLMs

## 🚀 Getting Started with Kernel Execution

### ⚙️ Supported Kernels and Short Names

| Kernel Description                  | Short Name |
| ----------------------------------- | ---------- |
| 🪞 FlatAttention (MHA, MQA, GQA)    | `attn`     |
| 🔢 GEMM with SUMMA Dataflow         | `gemm`     |
| 🌈 RMSNorm                          | `norm`     |
| ✨ Activation (Sigmoid, ReLU, SiLU) | `acti`     |
| Sparse DMA gather sanity test      | `sparse`   |
| Independent sparse attention access per cluster | `sparse_attn_access` |


---

### 🏗️ Build, Preload, and Run Simulation

To build configuration, prepare preload data, and run simulation, use:

```bash
make <Kernel>-run
```
The defualt kernel configuration are located in `config/kernels`

### Sparse DMA on SoftHier

Run `make sparse-runv` for the core-loop, inlined-loop and HW-gather cases with
HBM2E. Results are in `build/sparse_dma/`; see [SparseDMA](sw/SparseDMA/README.md)
for configuration and individual commands.

### Sparse attention access

Run `make sparse_attn_access-run` for all-cluster core-loop, inlined-loop and
HW-gather access to token-interleaved HBM. The default uses the west edge; a
16-node configuration enables all four edges. See
[SparseAttnAccess](sw/SparseAttnAccess/README.md) for configuration, runtime and
HBM utilization reports, and NoC traffic plots.

### FlooNoC v2

Set `SOFTHIER_DATA_NOC=floonoc_v2` for both build and simulation. The default is
`legacy`. V2 supports the sparse kernels with unaliased HBM nodes on any edge;
the synchronization network stays on the legacy model.

```bash
make sparse-runv SOFTHIER_DATA_NOC=floonoc_v2 \
  SPARSE_OUTPUT="$PWD/build/sparse_dma_noc_v2"

make sparse_attn_access-run SOFTHIER_DATA_NOC=floonoc_v2 \
  SPARSE_ATTN_ARCH_FILE="$PWD/config/arch/sparse_attn_access_hbm16.py" \
  SPARSE_ATTN_KERNEL_FILE="$PWD/config/kernels/sparse_attn_access_m1024_n128.py" \
  SPARSE_ATTN_OUTPUT="$PWD/build/sparse_attn_access_hbm16_m1024_n128_noc_v2"
```

Choose a fresh output directory for a new run. V2 traffic plots show request
routes and read-data return bandwidth separately. Backend selection, bridge
timing and supported operations are documented in
[DATA_NOC.md](../../pulp/pulp/chips/soft_hier_old/DATA_NOC.md).
