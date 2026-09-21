# SPDX-License-Identifier: Apache-2.0
SPARSE_ARCH_FILE ?= $(CUR_DIR)/config/arch/sparse_dma.py
SPARSE_KERNEL_FILE ?= $(CUR_DIR)/config/kernels/sparse_dma.py
SPARSE_VARIANTS ?= core-loop inlined-loop hw-gather
export SPARSE_ARCH_FILE SPARSE_KERNEL_FILE

.PHONY: sparse-cfg sparse-pld sparse-hw sparse-sw sparse-pre sparse-run sparse-runv sparse-report
.NOTPARALLEL: sparse-pre sparse-run sparse-runv

sparse-cfg sparse-pld:
	bash $(CUR_DIR)/scripts/sparse_dma/step.sh prepare

sparse-hw:
	bash $(CUR_DIR)/scripts/sparse_dma/step.sh hw

sparse-sw:
	@set -e; for variant in $(SPARSE_VARIANTS); do \
		bash $(CUR_DIR)/scripts/sparse_dma/step.sh sw $$variant; \
	done

sparse-pre: sparse-cfg sparse-hw sparse-sw

sparse-run sparse-runv: sparse-pre
	@set -e; for variant in $(SPARSE_VARIANTS); do \
		bash $(CUR_DIR)/scripts/sparse_dma/step.sh run $$variant; \
	done
	$(MAKE) sparse-report

sparse-report:
	bash $(CUR_DIR)/scripts/sparse_dma/step.sh report
