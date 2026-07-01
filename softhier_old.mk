#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chi Zhang <chizhang@ethz.ch>

######################################################################
## Legacy SoftHier target merged under pulp.pulp.chips.soft_hier_old ##
######################################################################

SOFTHIER_OLD_TARGET ?= pulp.chips.soft_hier_old.flex_cluster
SOFTHIER_OLD_CFG ?= soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py
SOFTHIER_OLD_APP ?= soft_hier_sdk/runtime/app_example
SOFTHIER_OLD_SW_BUILD ?= soft_hier_sdk/sw_build
SOFTHIER_OLD_CORE_MODEL ?= fast
SOFTHIER_OLD_PYTHON ?= python

ifdef cfg
SOFTHIER_OLD_CFG := $(cfg)
endif

ifdef app
SOFTHIER_OLD_APP := $(app)
endif

ifdef core_model
SOFTHIER_OLD_CORE_MODEL := $(core_model)
endif

SOFTHIER_OLD_TOOLCHAIN_BIN ?= $(CURDIR)/soft_hier_sdk/toolchain/install/bin

ifneq ($(SOFTHIER_OLD_TOOLCHAIN_BIN),)
export PATH := $(SOFTHIER_OLD_TOOLCHAIN_BIN):$(PATH)
endif

SOFTHIER_OLD_SW_CMAKE_ARG :=
ifdef app
SOFTHIER_OLD_SW_CMAKE_ARG := -DSRC_DIR=$(abspath $(SOFTHIER_OLD_APP))
endif

SOFTHIER_OLD_ARCH_CMAKE_ARG := $(shell if grep "spatz_attaced_core_list" "$(SOFTHIER_OLD_CFG)" | grep "\[\]" >/dev/null 2>&1; then echo "-DRISCV_ARCH=rv32imafd_zfh"; else echo "-DRISCV_ARCH=rv32imafdv_zfh"; fi)

SOFTHIER_OLD_PRELOAD_ARG :=
ifdef pld
SOFTHIER_OLD_PRELOAD_ARG := --preload $(abspath $(pld))
endif

.PHONY: sh-toolchain sh-old-config sh-old-hw sh-old-sw sh-old-hs sh-old-run sh-old-runv sh-old-pfto sh-old-clean-sw

# Downloads third-party toolchains under their upstream licenses:
# - pulp-riscv-gnu-toolchain v1.0.16: GNU toolchain components under
#   BSD-style and GNU GPL-family licenses.
# - husterZC/gun_toolchain v2.0.0: upstream repository has no declared
#   repository-level license; review notices in the downloaded archive.
$(CURDIR)/soft_hier_sdk/toolchain:
	mkdir -p $(CURDIR)/soft_hier_sdk/toolchain
	cd $(CURDIR)/soft_hier_sdk/toolchain && \
	wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 &&\
	tar -xvjf v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 &&\
	wget https://github.com/husterZC/gun_toolchain/releases/download/v2.0.0/toolchain.tar.xz &&\
	tar -xvf toolchain.tar.xz
sh-toolchain: $(CURDIR)/soft_hier_sdk/toolchain

sh-old-config:
	cp "$(SOFTHIER_OLD_CFG)" pulp/pulp/chips/soft_hier_old/flex_cluster_arch.py
	$(SOFTHIER_OLD_PYTHON) soft_hier_sdk/utilities/config.py "$(SOFTHIER_OLD_CFG)"

sh-old-hw: sh-old-config
	$(MAKE) TARGETS=$(SOFTHIER_OLD_TARGET) build

sh-old-sw: sh-old-config
	rm -rf "$(SOFTHIER_OLD_SW_BUILD)"
	mkdir -p "$(SOFTHIER_OLD_SW_BUILD)"
	cd "$(SOFTHIER_OLD_SW_BUILD)" && $(CMAKE) $(SOFTHIER_OLD_SW_CMAKE_ARG) $(SOFTHIER_OLD_ARCH_CMAKE_ARG) "$(abspath soft_hier_sdk/runtime)" && $(MAKE)
	@! grep -q "ebreak" "$(SOFTHIER_OLD_SW_BUILD)/softhier.dump" || (echo "Error: 'ebreak' found in $(SOFTHIER_OLD_SW_BUILD)/softhier.dump" && exit 1)

sh-old-hs: sh-old-hw sh-old-sw

sh-old-run:
	./install/bin/gvsoc --target=$(SOFTHIER_OLD_TARGET) --binary "$(SOFTHIER_OLD_SW_BUILD)/softhier.elf" $(SOFTHIER_OLD_PRELOAD_ARG) --core-model=$(SOFTHIER_OLD_CORE_MODEL) run --trace=/chip/cluster_0/redmule

sh-old-runv:
	./install/bin/gvsoc --target=$(SOFTHIER_OLD_TARGET) --binary "$(SOFTHIER_OLD_SW_BUILD)/softhier.elf" $(SOFTHIER_OLD_PRELOAD_ARG) --core-model=$(SOFTHIER_OLD_CORE_MODEL) run --trace=redmule --trace=/chip/cluster_.*/idma --trace=cluster_registers | tee "$(SOFTHIER_OLD_SW_BUILD)/analyze_trace.txt"

sh-old-pfto:
	$(SOFTHIER_OLD_PYTHON) soft_hier_sdk/utilities/trace_perfetto/parse.py "$(SOFTHIER_OLD_SW_BUILD)/analyze_trace.txt" "$(SOFTHIER_OLD_SW_BUILD)/roi.json"
	$(SOFTHIER_OLD_PYTHON) soft_hier_sdk/utilities/trace_perfetto/visualize.py "$(SOFTHIER_OLD_SW_BUILD)/roi.json" -o "$(SOFTHIER_OLD_SW_BUILD)/perfetto.json"

sh-old-clean-sw:
	rm -rf "$(SOFTHIER_OLD_SW_BUILD)"
