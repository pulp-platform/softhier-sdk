#!/usr/bin/env bash
# Source the workspace's GVSoC/DRAMSys environment and the documented SDK environment.
SPARSE_SDK_IMPL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SPARSE_GVSOC_ROOT="$(cd -- "$SPARSE_SDK_IMPL/../.." && pwd -P)"
source "$SPARSE_GVSOC_ROOT/build/reproduce_dramsys/env.sh"
cd "$SPARSE_GVSOC_ROOT"
source "$SPARSE_GVSOC_ROOT/soft_hier_sdk/sourceme.sh"
# Use the same HBM2E configuration and DRAMSys build as the standalone test.
export DRAMSYS_PATH="$SPARSE_GVSOC_ROOT/build/sparse_dma/dram"
export LD_LIBRARY_PATH="$SPARSE_GVSOC_ROOT/build/sparse_dma/dramsys-rtl/build/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$SPARSE_SDK_IMPL/build/sparse_dma/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$SPARSE_SDK_IMPL/build/sparse_dma/logs"
