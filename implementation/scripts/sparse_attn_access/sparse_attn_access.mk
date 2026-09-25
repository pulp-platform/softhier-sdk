# SPDX-License-Identifier: Apache-2.0
SPARSE_ATTN_ARCH_FILE ?= $(CUR_DIR)/config/arch/sparse_attn_access.py
SPARSE_ATTN_KERNEL_FILE ?= $(CUR_DIR)/config/kernels/sparse_attn_access.py
SPARSE_ATTN_OUTPUT ?= $(CUR_DIR)/build/sparse_attn_access
SPARSE_ATTN_BASELINE ?=
export SPARSE_ATTN_ARCH_FILE SPARSE_ATTN_KERNEL_FILE SPARSE_ATTN_OUTPUT SPARSE_ATTN_BASELINE

.PHONY: sparse_attn_access-cfg sparse_attn_access-hw sparse_attn_access-sw sparse_attn_access-pld sparse_attn_access-pre sparse_attn_access-run sparse_attn_access-runv sparse_attn_access-report
.NOTPARALLEL: sparse_attn_access-pre sparse_attn_access-run sparse_attn_access-runv

sparse_attn_access-cfg:
	bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh prepare

sparse_attn_access-hw:
	bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh hw

sparse_attn_access-sw:
	@set -e; for variant in core-loop inlined-loop hw-gather; do \
		bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh sw $$variant; \
	done

sparse_attn_access-pld:
	bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh preload

sparse_attn_access-pre: sparse_attn_access-cfg sparse_attn_access-hw sparse_attn_access-sw sparse_attn_access-pld

sparse_attn_access-run sparse_attn_access-runv: sparse_attn_access-pre
	@set -e; for variant in core-loop inlined-loop hw-gather; do \
		bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh run $$variant; \
	done
	$(MAKE) sparse_attn_access-report

sparse_attn_access-report:
	bash $(CUR_DIR)/scripts/sparse_attn_access/step.sh report
