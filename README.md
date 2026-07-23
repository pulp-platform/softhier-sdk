# 🌿 SoftHier SDK

The **SoftHier SDK** is a [PULP Platform](https://pulp-platform.org/index.html) software package developed by ETH Zurich and the University of Bologna as part of the [PULP project](https://pulp-platform.org/index.html).

This repository provides useful tools for programming and running software applications on **SoftHier**, a modeling and simulation framework for tile-based many-PE accelerators on the [GVSoC](https://github.com/gvsoc/gvsoc) event-based simulator. It is designed to be used together with a GVSoC source tree for benchmarking, debugging, and profiling software running on SoftHier architectures.

The SDK includes the SoftHier software runtime headers, startup and linker files, a default example application, configuration utilities, and Makefile integration targets for building and running SoftHier simulations.

For more background, see the [PULP Platform](https://pulp-platform.org/index.html) project page and the [GVSoC repository](https://github.com/gvsoc/gvsoc).

## 📍 Where This Fits

```text
gvsoc/
├── pulp/pulp/chips/soft_hier_old/     # Legacy SoftHier GVSoC models
├── soft_hier_sdk/                     # This SDK
│   ├── sourceme.sh                    # Environment + SDK-local toolchain setup
│   ├── softhier_old.mk                # Root make targets for SoftHier-old
│   ├── runtime/                       # Legacy runtime and linker scripts
│   ├── examples/SoftHier/             # Original example apps and arch configs
│   ├── implementation/                # Kernel-generation flows and LLM kernels
│   ├── utilities/                     # Arch/header/preload/perfetto utilities
└── Makefile                           # Includes soft_hier_sdk/softhier_old.mk
```

Run commands from the `gvsoc/` repository root unless a command explicitly uses
`make -C soft_hier_sdk/implementation`.

## ⚡ Quick Start

Navigate to the root of your GVSOC repository and add the following line to your `Makefile`:

```makefile
include soft_hier_sdk/softhier_old.mk
```

Initialize the SoftHier SDK environment:
```bash
source soft_hier_sdk/sourceme.sh
```

Then build and run the default SoftHier example:

```bash
make sh-old-hs
make sh-old-runv
```

The default target is:

```text
pulp.chips.soft_hier_old.flex_cluster
```

The default architecture config is:

```text
soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py
```

The default application is:

```text
soft_hier_sdk/runtime/app_example
```

## 🏗️ Root Make Targets

The root `Makefile` includes `soft_hier_sdk/softhier_old.mk`, which provides the legacy SoftHier
flow with explicit `sh-old-*` targets.

| Target | Purpose |
| --- | --- |
| `make sh-old-config` | Copy the selected arch config into the model tree and generate runtime arch headers. |
| `make sh-old-hw` | Build the merged GVSoC SoftHier-old hardware target. |
| `make sh-old-sw` | Build the selected SoftHier software app into `soft_hier_sdk/sw_build/softhier.elf`. |
| `make sh-old-hs` | Build both hardware and software. |
| `make sh-old-run` | Run with a focused RedMule trace for cluster 0. |
| `make sh-old-runv` | Run with RedMule, iDMA, and cluster-register traces, saving `analyze_trace.txt`. |
| `make sh-old-pfto` | Convert `analyze_trace.txt` to Perfetto JSON. |
| `make sh-old-clean-sw` | Remove `soft_hier_sdk/sw_build`. |

Useful make variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `cfg=...` | `soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py` | Architecture configuration. |
| `app=...` | `soft_hier_sdk/runtime/app_example` | Software application directory. |
| `core_model=fast` | `fast` | Core model selector: `fast` or `accurate`. |
| `pld=...` | empty | Optional HBM preload ELF. |

Example:

```bash
make sh-old-hs \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  app=soft_hier_sdk/examples/SoftHier/software/gemm_systolic \
  core_model=fast

make sh-old-runv \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  core_model=fast
```

## 🧠 Core Model Selection

SoftHier-old can run with either:

| Mode | Use |
| --- | --- |
| `core_model=fast` | Fast Snitch model. This is the default and the recommended mode for long legacy apps. |
| `core_model=accurate` | More detailed Snitch + FP subsystem path. Use for core-model debugging. |

The selector is passed to GVSoC as:

```text
--core-model=fast
```

or:

```text
--core-model=accurate
```

## ✅ Verified Applications

### 🏷️ Annotation API Test

The annotation test exercises the complete annotation lifecycle on every core,
including concurrent IDs, dynamic labels, invalid-label rejection, active
interval handling, ID `0`, `UINT32_MAX`, and per-cluster isolation.

Build the hardware model and test application:

```bash
source soft_hier_sdk/sourceme.sh

make sh-old-hs \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC1024.py \
  app=soft_hier_sdk/examples/SoftHier/software/annotation_test \
  core_model=fast
```

Run with cluster-register tracing enabled:

```bash
make sh-old-runv \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC1024.py \
  core_model=fast
```

Check the application result:

```bash
rg "ANNOTATION_TEST_(PASS|FAIL)" soft_hier_sdk/sw_build/analyze_trace.txt
```

A successful run prints:

```text
ANNOTATION_TEST_PASS
```

Convert the annotation intervals into cluster-scoped Perfetto tracks:

```bash
make sh-old-pfto
```

The verbose trace and converted output are written to:

```text
soft_hier_sdk/sw_build/analyze_trace.txt
soft_hier_sdk/sw_build/roi.json
soft_hier_sdk/sw_build/perfetto.json
```

### 🟥 GEMM Systolic, NoC512

```bash
source soft_hier_sdk/sourceme.sh

make sh-old-hs \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  app=soft_hier_sdk/examples/SoftHier/software/gemm_systolic \
  core_model=fast

make sh-old-runv \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  core_model=fast
```

RedMule checkpoint:

```bash
rg -c "GEMM id" soft_hier_sdk/sw_build/analyze_trace.txt
rg "GEMM id = 256|Performance Counter|Simulation stopped" soft_hier_sdk/sw_build/analyze_trace.txt
```

A healthy full run should frequently print lines like:

```text
[LightRedmule] Finished : ... | GEMM id = ... | M-N-K = 256-256-256
```

### 🟥 GEMM Systolic, NoC1024

```bash
source soft_hier_sdk/sourceme.sh

make sh-old-hs \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC1024.py \
  app=soft_hier_sdk/examples/SoftHier/software/gemm_systolic \
  core_model=fast

make sh-old-runv \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC1024.py \
  core_model=fast
```

Use the same RedMule checkpoint commands as NoC512.

### 🟦 RMSNorm

The RMSNorm flow lives under `soft_hier_sdk/implementation` because it generates kernel-specific
headers and preload data before building the old app.

```bash
source soft_hier_sdk/sourceme.sh
make -C soft_hier_sdk/implementation norm-runv
```

This uses:

```text
soft_hier_sdk/implementation/config/arch/arch.py
soft_hier_sdk/implementation/config/kernels/norm.py
soft_hier_sdk/implementation/sw/RMSNorm
```

Generated files:

```text
soft_hier_sdk/implementation/sw/RMSNorm/include/norm.h
soft_hier_sdk/implementation/sw/RMSNorm/include/preload.h
soft_hier_sdk/implementation/sw/RMSNorm/preload.elf
```

RMSNorm mainly exercises iDMA and vector instructions, not the RedMule GEMM datapath.

## 🧪 Trace And Perfetto Flow

Run a verbose trace:

```bash
make sh-old-runv \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  core_model=fast
```

Trace output:

```text
soft_hier_sdk/sw_build/analyze_trace.txt
```

Convert to Perfetto:

```bash
make sh-old-pfto
```

Generated files:

```text
soft_hier_sdk/sw_build/roi.json
soft_hier_sdk/sw_build/perfetto.json
```

## 🧬 Architecture And Kernel Configuration

Architecture configs define cluster count, core count, TCDM, HBM layout, NoC width, Spatz attachment,
RedMule shape, and synchronization ranges.

Common configs:

```text
soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py
soft_hier_sdk/examples/SoftHier/config/arch_NoC1024.py
soft_hier_sdk/implementation/config/arch/arch.py
```

Kernel configs for the implementation flow:

```text
soft_hier_sdk/implementation/config/kernels/attn.py
soft_hier_sdk/implementation/config/kernels/gemm.py
soft_hier_sdk/implementation/config/kernels/norm.py
soft_hier_sdk/implementation/config/kernels/acti.py
```

The implementation Makefile supports:

```bash
make -C soft_hier_sdk/implementation attn-run
make -C soft_hier_sdk/implementation gemm-run
make -C soft_hier_sdk/implementation norm-run
make -C soft_hier_sdk/implementation acti-run
```

Use `*-runv` for verbose traces.

## 🧹 Cleaning

Remove the root software build:

```bash
make sh-old-clean-sw
```

Remove generated implementation-kernel files:

```bash
make -C soft_hier_sdk/implementation norm-clean
make -C soft_hier_sdk/implementation gemm-clean
make -C soft_hier_sdk/implementation attn-clean
make -C soft_hier_sdk/implementation acti-clean
```

The SDK-local toolchain is intentionally not removed by these commands.

## 🛠️ Troubleshooting

### Compiler is not SDK-local

Check:

```bash
source soft_hier_sdk/sourceme.sh
which riscv32-unknown-elf-gcc
```

Expected:

```text
.../soft_hier_sdk/toolchain/install/bin/riscv32-unknown-elf-gcc
```

If the compiler is missing, check that the old-release archive exists:

```text
../soft_hier_release/third_party/toolchain/toolchain.tar.xz
```

### RedMule trace is silent

For GEMM, the correct run should frequently emit:

```text
[LightRedmule] Finished : ... GEMM id = ...
```

If not, verify:

1. The app is `soft_hier_sdk/examples/SoftHier/software/gemm_systolic`.
2. The command used `make sh-old-runv` or explicitly enabled a RedMule trace.
3. The architecture config was copied through `make sh-old-config` or any target depending on it.
4. The binary was rebuilt after changing `cfg` or `app`.

### Build uses stale software

Rebuild software explicitly:

```bash
make sh-old-sw \
  cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
  app=soft_hier_sdk/examples/SoftHier/software/gemm_systolic
```

### RMSNorm headers are missing

Use the implementation Makefile; it generates `norm.h`, `preload.h`, and `preload.elf` before build:

```bash
make -C soft_hier_sdk/implementation norm-pre
```

## 📌 Notes For Maintainers

- Keep old application source code unchanged when validating the merge.
- Keep SoftHier-old hardware compatibility changes inside `pulp/pulp/chips/soft_hier_old`.
- `soft_hier_sdk/toolchain/` is generated by `sourceme.sh` and ignored by git.
- `soft_hier_sdk/sw_build/` is a generated software build directory.
- For regression checks, GEMM RedMule trace density is the strongest signal that the old custom
  instruction path, offload decoder, iDMA, RedMule, and synchronization path are all connected.
