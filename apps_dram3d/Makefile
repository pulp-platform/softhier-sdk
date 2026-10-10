# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
APP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SDK_DIR := $(abspath $(APP_DIR)/..)
GVSOC_DIR := $(abspath $(SDK_DIR)/..)
RUNTIME := $(SDK_DIR)/runtime/runtime
ARCH ?= $(APP_DIR)/config/arch.py
CASE ?= big1024
DATAFLOW ?= auto
SEED ?= 19
PYTHON ?= python3
CC = $(SDK_DIR)/toolchain/install/bin/riscv32-unknown-elf-gcc
GVSOC ?= $(GVSOC_DIR)/install/bin/gvsoc
OUT ?= $(APP_DIR)/build/$(CASE)-$(DATAFLOW)
TIMEOUT ?= 600
TRACE_REDMULE ?= 0
CASES := big1024 big2048 flat16 flat8

.PHONY: all hw generate build run suite
all: build

# Select this architecture without copying it over the hardware default.
hw:
	SOFTHIER_ARCH_FILE=$(abspath $(ARCH)) $(MAKE) -C $(GVSOC_DIR) \
		TARGETS=pulp.chips.soft_hier_old.flex_cluster build

generate:
	$(PYTHON) $(SDK_DIR)/utilities/config.py $(ARCH) --output-dir $(OUT)
	$(PYTHON) $(APP_DIR)/scripts/generate.py --arch $(ARCH) --case $(CASE) \
		--dataflow $(DATAFLOW) --seed $(SEED) --output $(OUT)

build: generate
	$(CC) -I $(OUT) -I $(RUNTIME)/include -T $(OUT)/memory.ld \
		-nostartfiles -mabi=ilp32d -march=rv32imafd_zfh -mcmodel=medlow \
		-O3 -g -fno-builtin -fno-tree-vectorize -ffunction-sections -fdata-sections \
		-Wl,--gc-sections $(RUNTIME)/flex_start.s $(APP_DIR)/sw/gemm/main.c -o $(OUT)/gemm.elf

run: build
	$(PYTHON) $(APP_DIR)/scripts/run.py --gvsoc $(GVSOC) --arch $(ARCH) \
		--output $(OUT) --timeout $(TIMEOUT) $(if $(filter 1,$(TRACE_REDMULE)),--trace-redmule)

suite:
	@set -e; for case in $(CASES); do $(MAKE) run CASE=$$case; done
