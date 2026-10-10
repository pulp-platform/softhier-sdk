# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
APP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SDK_DIR := $(abspath $(APP_DIR)/../..)
GVSOC_DIR := $(abspath $(SDK_DIR)/..)
RUNTIME := $(SDK_DIR)/runtime/runtime
ARCH ?= $(SDK_DIR)/apps_dram3d/config/arch.py
CASE ?= decode_b1_kv1024
LAYER ?= 1
SEED ?= 19
PYTHON ?= python3
CC = $(SDK_DIR)/toolchain/install/bin/riscv32-unknown-elf-gcc
GVSOC ?= $(GVSOC_DIR)/install/bin/gvsoc
OUT ?= $(APP_DIR)/build/$(CASE)-l$(LAYER)-s$(SEED)
TIMEOUT ?= 3600
ALL_EXPERTS ?= 0
EXTRA_CFLAGS ?=
DECODE_CASES := $(foreach b,1 8 16,$(foreach k,1024 2048 4096,decode_b$(b)_kv$(k)))
PREFILL_CASES := prefill_s64 prefill_s128
CASES ?= $(DECODE_CASES) $(PREFILL_CASES)
GEN_ARGS = --arch $(ARCH) --case $(CASE) --layer $(LAYER) \
           --seed $(SEED) --output $(OUT) $(if $(filter 1,$(ALL_EXPERTS)),--all-experts)

.PHONY: all plan generate compile build run simulate suite
all: build

plan:
	$(PYTHON) $(APP_DIR)/scripts/generate.py $(GEN_ARGS) --plan-only

generate:
	$(PYTHON) $(SDK_DIR)/utilities/config.py $(ARCH) --output-dir $(OUT)
	$(PYTHON) $(APP_DIR)/scripts/generate.py $(GEN_ARGS)

# compile lets a developer rebuild C without regenerating the large preload.
compile:
	$(CC) -I $(OUT) -isystem $(RUNTIME)/include -I $(APP_DIR)/sw -T $(OUT)/memory.ld \
		-nostartfiles -mabi=ilp32d -march=rv32imafdv_zfh_zvfh -mcmodel=medlow \
		-O3 -g -Wall -Wextra -fno-builtin -fno-tree-vectorize -ffp-contract=off $(EXTRA_CFLAGS) \
		-ffunction-sections -fdata-sections -Wl,--gc-sections \
		$(RUNTIME)/flex_start.s $(APP_DIR)/sw/main.c -o $(OUT)/layer.elf

build: generate
	$(MAKE) -f $(APP_DIR)/Makefile compile OUT=$(OUT)

simulate:
	$(PYTHON) $(APP_DIR)/scripts/run.py --gvsoc $(GVSOC) --arch $(ARCH) \
		--output $(OUT) --timeout $(TIMEOUT)

run: build
	$(MAKE) -f $(APP_DIR)/Makefile simulate OUT=$(OUT)

suite:
	@set -e; for case in $(CASES); do $(MAKE) -f $(APP_DIR)/Makefile run CASE=$$case; done
