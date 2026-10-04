# Reproducing the simulator environment

The SDK checkout belongs at `gvsoc/soft_hier_sdk`. Run the commands below from
the GVSOC root. The decoder itself is built from the SDK runtime; it does not
require the SDK's CMake application generator.

The evaluated simulator revisions are:

| Repository | Branch | Commit |
| --- | --- | --- |
| `gvsoc/gvsoc` | `chi/power_interface` | `0ab0442c07d030d380010c59d3004d5a5b7c7f3f` |
| `gvsoc/gvsoc-core` | `chi/power_interface` | `4be5293a4976eea02d224e425493b755d46b1d15` |
| `gvsoc/gvsoc-pulp` | `chi/update_soft_hier` | `f822a9413950f713159f85c4dd1509963e151b4e` |
| `husterZC/gvsoc-engine` | `chi/power_interface` | `16dadd30d76a0aa2a33a85d3bdc52a75558a83c3` |
| `pulp-platform/softhier-sdk` | `chi/soft_hier_old_llm_map` | decoder source: `42313371d6219e29308babc9dad6812fc8244efd` |

Initialize the root's submodules at their recorded revisions. Clone the SDK
separately if it is absent:

```sh
git submodule update --init --recursive
git clone --branch chi/soft_hier_old_llm_map \
  git@github.com:pulp-platform/softhier-sdk.git soft_hier_sdk
```

## Dependencies

The evaluation used Python 3.12, NumPy, CMake 3.31.10, host GCC/G++ 8, and the
SDK's RISC-V GCC 14.2 toolchain. Python must be at least 3.11 because the runner
uses `hashlib.file_digest`. Install the root/core Python requirements and NumPy
in a local environment such as `build/venv`. Put the RISC-V toolchain at
`third_party/softhier-toolchain/install`, or adjust the PATH below. Its
`riscv32-unknown-elf-gcc` must support `rv32imafdv_zfh`.

SystemC and DRAMSys must use compatible host compilers and the same C++ standard.
The evaluated SystemC version is **2.3.3, built with C++17**. With its source at
`third_party/systemc-2.3.3`:

```sh
build/venv/bin/cmake -S third_party/systemc-2.3.3 -B third_party/systemc-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_INSTALL_PREFIX="$PWD/third_party/systemc_install" \
  -DCMAKE_INSTALL_LIBDIR=lib64
build/venv/bin/cmake --build third_party/systemc-build -j 8
build/venv/bin/cmake --install third_party/systemc-build
```

Build DRAMSys from commit **8565f18**, with the root repository's
`add_dramsyslib_patches/build_dynlib_from_github_dramsys5/patch` applied.
The bundled shared library uses an older interface and cannot substitute for
this rebuild. Three compatibility edits are also needed for the evaluated
SystemC 2.3.3/GCC 8 combination:

- In `apps/simulator/simulator/request/RequestIssuer.h`, add
  `SC_HAS_PROCESS(RequestIssuer);` in the public section of `RequestIssuer`.
- In `apps/simulator/CMakeLists.txt`, add `simulator/elfloader.cpp` to the
  `simulator` library's sources.
- Add `stdc++fs` to that library's `target_link_libraries` list for GCC 8.

With this prepared source at `third_party/DRAMSys-8565f18`:

```sh
LIBRARY_PATH= LD_LIBRARY_PATH= build/venv/bin/cmake \
  -S third_party/DRAMSys-8565f18 -B third_party/dramsys-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC \
  -DDRAMSYS_USE_FETCH_CONTENT_SYSTEMC=OFF \
  -DSystemCLanguage_DIR="$PWD/third_party/systemc_install/lib64/cmake/SystemCLanguage"
LIBRARY_PATH= LD_LIBRARY_PATH= build/venv/bin/cmake \
  --build third_party/dramsys-build -j 8
```

The resulting library is
`third_party/dramsys-build/lib/libDRAMSys_Simulator.so`. The simulator wrapper
requires its `add_dram` memory-specification output and byte preload/read APIs.

## HBM configuration

Copy the tracked configurations into a run-local directory. The configured
pseudo-channel is 32 bits wide, so byte selection uses address bits 0 and 1.
Removing the original byte bit 2 shifts every higher address-mapping bit down
one. This gives 1 GiB of address coverage per physical channel. Storage must be
enabled for functional execution; database recording is disabled in this run.

The following is repeatable: each invocation starts from the tracked configs.
Use a fresh destination when another simulation is running with a different
configuration.

```sh
build/venv/bin/python - <<'PY'
import json
from pathlib import Path
import shutil

dst = Path('build/gpt_oss_decode/environment/dramsys_configs')
shutil.copytree('add_dramsyslib_patches/dramsys_configs', dst, dirs_exist_ok=True)
p = dst / 'addressmapping/am_hbm4_emu_16Gb_pc_brc.json'
obj = json.loads(p.read_text())
for key, bits in obj['addressmapping'].items():
    obj['addressmapping'][key] = [b if b < 2 else b - 1 for b in bits if b != 2]
p.write_text(json.dumps(obj, indent=2) + '\n')
p = dst / 'simconfig/example.json'
obj = json.loads(p.read_text())
obj['simconfig'].update(StoreMode='Store', DatabaseRecording=False)
p.write_text(json.dumps(obj, indent=2) + '\n')
PY
```

## Build and execute

The paths below correspond to the local dependency layout above. Set the
environment in each new shell, from the GVSOC root:

```sh
export PATH="$PWD/build/venv/bin:$PWD/install/bin:$PWD/third_party/softhier-toolchain/install/bin:/usr/bin:/bin"
export PYTHONPATH="$PWD/install/python:$PWD/install/generators"
export SYSTEMC_HOME="$PWD/third_party/systemc_install"
export LD_LIBRARY_PATH="$PWD/install/lib:$PWD/third_party/systemc_install/lib64:$PWD/third_party/dramsys-build/lib"
export DRAMSYS_PATH="$PWD/build/gpt_oss_decode/environment"
export TMPDIR="$PWD/build/gpt_oss_decode/tmp"
export SOFTHIER_ARCH_FILE="$PWD/soft_hier_sdk/LLM_decode/gpt-oss-120b/config/arch.py"
mkdir -p "$TMPDIR"

CCACHE_DISABLE=1 CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  make build TARGETS=pulp.chips.soft_hier_old.flex_cluster \
  CMAKE="$PWD/build/venv/bin/cmake" CMAKE_FLAGS='-j 8'

make -C soft_hier_sdk/LLM_decode/gpt-oss-120b regression
make -C soft_hier_sdk/LLM_decode/gpt-oss-120b smoke
make -C soft_hier_sdk/LLM_decode/gpt-oss-120b benchmark
```

In the original evaluation checkout, `source build/direct_preload/env.sh`
selects the same libraries and HBM configuration from their existing locations.
That local convenience file is not required when using the setup above.

All batches use the same `decode.elf` and `weights.elf`. Check
`build/gpt_oss_decode/full/RESULTS.md` for the measured stage breakdown and
`batch-N/validation.json` for numerical checks. Simulation has a six-hour host
timeout per batch; reported layer latency is simulated time and excludes ELF
initialization and output dumping.
