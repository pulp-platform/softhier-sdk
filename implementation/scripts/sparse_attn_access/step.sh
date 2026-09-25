#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../sparse_dma/env.sh"
SAA_OUTPUT="${SPARSE_ATTN_OUTPUT:-$SPARSE_SDK_IMPL/build/sparse_attn_access}"
SAA_ARCH="${SPARSE_ATTN_ARCH_FILE:-$SPARSE_SDK_IMPL/config/arch/sparse_attn_access.py}"
SAA_KERNEL="${SPARSE_ATTN_KERNEL_FILE:-$SPARSE_SDK_IMPL/config/kernels/sparse_attn_access.py}"
SAA_APP="$SPARSE_SDK_IMPL/sw/SparseAttnAccess"
SAA_VARIANT="${2:-core-loop}"
case "$SAA_VARIANT" in
    core-loop) SAA_APP_VARIANT=core_loop ;;
    inlined-loop) SAA_APP_VARIANT=inlined_loop ;;
    hw-gather) SAA_APP_VARIANT=hw_gather ;;
    *) echo "Unknown case: $SAA_VARIANT" >&2; exit 2 ;;
esac
mkdir -p "$SAA_OUTPUT/logs"
case "${1:-}" in
    prepare)
        python3 "$SPARSE_SDK_IMPL/scripts/kernels/sparse_attn_access_preload.py" \
            prepare "$SAA_ARCH" "$SAA_KERNEL" "$SAA_APP" "$SAA_OUTPUT"
        ;;
    hw)
        make -C "$SPARSE_GVSOC_ROOT" sh-old-hw cfg="$SAA_ARCH" \
            SOFTHIER_OLD_PYTHON=python3 CMAKE_FLAGS='-j 8' \
            > "$SAA_OUTPUT/logs/hw-build.log" 2>&1
        ;;
    sw)
        mkdir -p "$SAA_OUTPUT/$SAA_VARIANT"
        make -C "$SPARSE_GVSOC_ROOT" sh-old-sw cfg="$SAA_ARCH" \
            app="$SAA_APP/$SAA_APP_VARIANT" SOFTHIER_OLD_PYTHON=python3 \
            SOFTHIER_OLD_SW_BUILD="$SAA_OUTPUT/$SAA_VARIANT/sw" \
            > "$SAA_OUTPUT/$SAA_VARIANT/sw-build.log" 2>&1
        python3 "$SPARSE_SDK_IMPL/scripts/sparse_dma/check_elf.py" \
            "$SAA_OUTPUT/$SAA_VARIANT/sw/softhier.elf" "$SAA_OUTPUT/workload.json"
        ;;
    preload)
        python3 "$SPARSE_SDK_IMPL/scripts/kernels/sparse_attn_access_preload.py" \
            preload "$SAA_ARCH" "$SAA_KERNEL" "$SAA_APP" "$SAA_OUTPUT"
        ;;
    run)
        python3 "$SPARSE_SDK_IMPL/scripts/sparse_attn_access/run.py" "$SAA_OUTPUT" "$SAA_VARIANT"
        ;;
    report)
        SAA_REPORT_ARGS=("$SAA_OUTPUT")
        if [ -n "${SPARSE_ATTN_BASELINE:-}" ]; then
            SAA_REPORT_ARGS+=(--baseline "$SPARSE_ATTN_BASELINE")
        fi
        python3 "$SPARSE_SDK_IMPL/scripts/sparse_attn_access/analyze.py" "${SAA_REPORT_ARGS[@]}"
        ;;
    *) echo "Usage: $0 {prepare|hw|sw CASE|preload|run CASE|report}" >&2; exit 2 ;;
esac
