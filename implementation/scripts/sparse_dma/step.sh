#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SPARSE_ARCH_FILE="${SPARSE_ARCH_FILE:-$SPARSE_SDK_IMPL/config/arch/sparse_dma.py}"
SPARSE_KERNEL_FILE="${SPARSE_KERNEL_FILE:-$SPARSE_SDK_IMPL/config/kernels/sparse_dma.py}"
SPARSE_APP_DIR="$SPARSE_SDK_IMPL/sw/SparseDMA"
SPARSE_OUTPUT_DIR="$SPARSE_SDK_IMPL/build/sparse_dma"
SPARSE_VARIANT="${2:-core-loop}"
case "$SPARSE_VARIANT" in
    core-loop) SPARSE_APP_VARIANT=core_loop ;;
    inlined-loop) SPARSE_APP_VARIANT=inlined_loop ;;
    hw-gather) SPARSE_APP_VARIANT=hw_gather ;;
    *) echo "Unknown sparse DMA variant: $SPARSE_VARIANT" >&2; exit 2 ;;
esac
SPARSE_VARIANT_DIR="$SPARSE_OUTPUT_DIR/$SPARSE_VARIANT"
case "${1:-}" in
    prepare)
        python3 "$SPARSE_SDK_IMPL/scripts/kernels/sparse_dma_preload.py" \
            "$SPARSE_ARCH_FILE" "$SPARSE_KERNEL_FILE" "$SPARSE_APP_DIR" "$SPARSE_OUTPUT_DIR" \
            > "$SPARSE_OUTPUT_DIR/logs/prepare.log" 2>&1
        tail -n 5 "$SPARSE_OUTPUT_DIR/logs/prepare.log"
        ;;
    hw)
        make -C "$SPARSE_GVSOC_ROOT" sh-old-hw cfg="$SPARSE_ARCH_FILE" \
            SOFTHIER_OLD_PYTHON=python3 CMAKE_FLAGS='-j 8' \
            > "$SPARSE_OUTPUT_DIR/logs/hw-build.log" 2>&1
        ;;
    sw)
        mkdir -p "$SPARSE_VARIANT_DIR"
        make -C "$SPARSE_GVSOC_ROOT" sh-old-sw cfg="$SPARSE_ARCH_FILE" \
            app="$SPARSE_APP_DIR/$SPARSE_APP_VARIANT" SOFTHIER_OLD_PYTHON=python3 \
            SOFTHIER_OLD_SW_BUILD="$SPARSE_VARIANT_DIR/sw" \
            > "$SPARSE_VARIANT_DIR/sw-build.log" 2>&1
        # Enforce the actual 64 KiB instruction-memory and 384 KiB local-memory limits.
        python3 "$SPARSE_SDK_IMPL/scripts/sparse_dma/check_elf.py" \
            "$SPARSE_VARIANT_DIR/sw/softhier.elf" "$SPARSE_OUTPUT_DIR/workload.json"
        ;;
    run)
        mkdir -p "$SPARSE_VARIANT_DIR/run"
        cd "$SPARSE_VARIANT_DIR/run"
        set +e
        timeout 300 "$SPARSE_GVSOC_ROOT/install/bin/gvsoc" \
            --target=pulp.chips.soft_hier_old.flex_cluster \
            --binary="$SPARSE_VARIANT_DIR/sw/softhier.elf" \
            --preload="$SPARSE_APP_DIR/preload.elf" --core-model=fast \
            run --trace=/chip/cluster_0/idma --trace=/chip/data_noc/ni_1_1 \
            --trace=/chip/data_noc/router_1_1 --trace=/chip/west_hbm_ctrl \
            --trace-level=trace > simulation.log 2>&1
        SPARSE_RUN_STATUS=$?
        set -e
        printf '%s\n' "$SPARSE_RUN_STATUS" > exit-status.txt
        # The existing SoftHier EOC register ignores the software return value.
        # Require an explicit application PASS as well as simulator exit status.
        if [ "$SPARSE_RUN_STATUS" -ne 0 ] || \
           ! grep -q "SPARSE_DMA_PASS variant=$SPARSE_VARIANT" simulation.log || \
           grep -q 'SPARSE_DMA_FAIL\|SPARSE_DMA_MISMATCH' simulation.log; then
            tail -n 30 simulation.log
            echo "Sparse DMA simulation failed; see $SPARSE_VARIANT_DIR/run/simulation.log" >&2
            exit 1
        fi
        grep 'SPARSE_DMA_' simulation.log
        ;;
    report)
        python3 "$SPARSE_SDK_IMPL/scripts/sparse_dma/analyze.py" "$SPARSE_OUTPUT_DIR"
        ;;
    *) echo "Usage: $0 {prepare|hw|sw VARIANT|run VARIANT|report}" >&2; exit 2 ;;
esac
