# 🌿 SoftHier SDK

The **SoftHier SDK** is a [PULP Platform](https://pulp-platform.org/index.html) software package developed by ETH Zurich and the University of Bologna as part of the [PULP project](https://pulp-platform.org/index.html).

This repository provides useful tools for programming and running software applications on **SoftHier**, a modeling and simulation framework for tile-based many-PE accelerators on the [GVSoC](https://github.com/gvsoc/gvsoc) event-based simulator. It is designed to be used together with a GVSoC source tree for benchmarking, debugging, and profiling software running on SoftHier architectures.

The SDK includes the SoftHier software runtime headers, startup and linker files, a default example application, configuration utilities, and Makefile integration targets for building and running SoftHier simulations.

For more background, see the [PULP Platform](https://pulp-platform.org/index.html) project page and the [GVSoC repository](https://github.com/gvsoc/gvsoc).

---

## 🚀 Getting Started

Follow these steps to set up and run your first SoftHier simulation:

### 1️⃣ Set up the environment

Initialize the SoftHier SDK environment:

```bash
source sourceme.sh
```

---

### 2️⃣ Integrate with GVSOC

Navigate to the root of your GVSOC repository and add the following line to your `Makefile`:

```makefile
include $(SOFTHIER_SDK)/softhier.mk
```

---

### 3️⃣ Install toolchains

Install all required third-party toolchains:

```bash
make third_party/toolchain
```

This target downloads and extracts the following third-party archives into `third_party/toolchain`:

- `v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2` from `pulp-platform/pulp-riscv-gnu-toolchain`. This is a prebuilt RISC-V GNU compiler toolchain. Its upstream license file covers multiple components, including newlib under BSD-style terms and GCC/binutils/Linux headers under GNU GPL-family terms.
- `toolchain.tar.xz` from `husterZC/gun_toolchain` release `v2.0.0`. This is a prebuilt GNU/RISC-V toolchain with vector and Float16 support. The upstream GitHub repository does not declare a repository-level license; review the license notices included in the downloaded archive before redistribution.

These toolchains are not part of the SoftHier SDK license grant and remain under their respective upstream licenses.

---

### 4️⃣ Build SoftHier hardware 🛠️

```bash
make sh-hw
```

---

### 5️⃣ Build SoftHier software 💻

```bash
make sh-sw
```

📂 The default example application is located at:

```
<path/to/softhier-sdk>/sw/app_example
```

---

### 6️⃣ Run the simulation ▶️

```bash
make sh-run
```

---

## 📜 License

Copyright ETH Zurich and University of Bologna 2026.

The SoftHier SDK is licensed under the **Apache v2.0 license**. See the [`LICENSE`](./LICENSE) file for full details.
