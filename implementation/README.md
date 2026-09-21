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
