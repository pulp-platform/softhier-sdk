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

CUR_DIR := $(shell pwd)
SH_DIR := $(realpath ../..)

ARCH_FILE ?= ${CUR_DIR}/config/arch/arch.py
ifdef arch
	ARCH_FILE = $(realpath $(arch))
endif

export PYTHONPATH := ${CUR_DIR}/scripts:$(PYTHONPATH)


hw:
	@echo "Compiling SoftHier HW & SW for attention ..."
	cfg=${ARCH_FILE} make -C ${SH_DIR} sh-old-hw

pfto:
	@echo "Compiling SoftHier HW & SW for attention ..."
	make -C ${SH_DIR} sh-old-pfto

kernel-clean: attn-clean gemm-clean norm-clean acti-clean
clean: kernel-clean

#####################
#   FlatAttention   #
#####################

ATTN_FILE ?= ${CUR_DIR}/config/kernels/attn.py
ifdef attn
	ATTN_FILE = $(realpath $(attn))
endif

attn-cfg:
	@echo "Generating configuration files for attention ..."
	python3 ${CUR_DIR}/scripts/kernels/attn_config.py ${ATTN_FILE} ${CUR_DIR}/sw/FlatAttention/include/attn.h

attn-pld:
	@echo "Generating preload files for attention ..."
	python3 ${CUR_DIR}/scripts/kernels/attn_preload.py \
			${CUR_DIR}/sw/FlatAttention/preload.elf \
			${CUR_DIR}/sw/FlatAttention/include/preload.h \
			${ARCH_FILE} \
			${ATTN_FILE}

attn-hs:
	@echo "Compiling SoftHier HW & SW for attention ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/FlatAttention make -C ${SH_DIR} sh-old-hs

attn-sw:
	@echo "Compiling SoftHier HW & SW for attention ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/FlatAttention make -C ${SH_DIR} sh-old-sw

attn-pre: attn-clean attn-cfg attn-pld attn-hs

attn-run: attn-pre
	@echo "SoftHier simulation for attention ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/FlatAttention pld=${CUR_DIR}/sw/FlatAttention/preload.elf make -C ${SH_DIR} sh-old-run

attn-run%: attn-pre
	@echo "SoftHier simulation for attention ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/FlatAttention pld=${CUR_DIR}/sw/FlatAttention/preload.elf make -C ${SH_DIR} sh-old-run$*

attn-clean:
	@echo "clean up intermidiate files generated for attention ..."
	rm -f ${CUR_DIR}/sw/FlatAttention/include/attn.h ${CUR_DIR}/sw/FlatAttention/preload.elf ${CUR_DIR}/sw/FlatAttention/preload.dump ${CUR_DIR}/sw/FlatAttention/include/preload.h ${SH_DIR}/dump_*



#############
#   GEMM    #
#############

GEMM_FILE ?= ${CUR_DIR}/config/kernels/gemm.py
ifdef gemm
	GEMM_FILE = $(realpath $(gemm))
endif

gemm-cfg:
	@echo "Generating configuration files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_config.py ${GEMM_FILE} ${CUR_DIR}/sw/SummaGEMM/include/gemm.h

gemm-pld:
	@echo "Generating preload files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_preload.py \
			${CUR_DIR}/sw/SummaGEMM/preload.elf \
			${CUR_DIR}/sw/SummaGEMM/include/preload.h \
			${ARCH_FILE} \
			${GEMM_FILE}

gemm-hs:
	@echo "Compiling SoftHier HW & SW for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/SummaGEMM make -C ${SH_DIR} sh-old-hs

gemm-sw:
	@echo "Compiling SoftHier HW & SW for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/SummaGEMM make -C ${SH_DIR} sh-old-sw

gemm-pre: gemm-clean gemm-cfg gemm-pld gemm-hs

gemm-run: gemm-pre
	@echo "SoftHier simulation for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/SummaGEMM pld=${CUR_DIR}/sw/SummaGEMM/preload.elf make -C ${SH_DIR} sh-old-run

gemm-run%: gemm-pre
	@echo "SoftHier simulation for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/SummaGEMM pld=${CUR_DIR}/sw/SummaGEMM/preload.elf make -C ${SH_DIR} sh-old-run$*

gemm-clean:
	@echo "clean up intermidiate files generated for GEMM ..."
	rm -f ${CUR_DIR}/sw/SummaGEMM/include/gemm.h ${CUR_DIR}/sw/SummaGEMM/preload.elf ${CUR_DIR}/sw/SummaGEMM/preload.dump ${CUR_DIR}/sw/SummaGEMM/include/preload.h ${SH_DIR}/dump_*



######################
#   Normolization    #
######################

NORM_FILE ?= ${CUR_DIR}/config/kernels/norm.py
ifdef norm
	NORM_FILE = $(realpath $(norm))
endif

norm-cfg:
	@echo "Generating configuration files for Normolization ..."
	python3 ${CUR_DIR}/scripts/kernels/norm_config.py ${NORM_FILE} ${CUR_DIR}/sw/RMSNorm/include/norm.h

norm-pld:
	@echo "Generating preload files for Normolization ..."
	python3 ${CUR_DIR}/scripts/kernels/norm_preload.py \
			${CUR_DIR}/sw/RMSNorm/preload.elf \
			${CUR_DIR}/sw/RMSNorm/include/preload.h \
			${ARCH_FILE} \
			${NORM_FILE}

norm-hs:
	@echo "Compiling SoftHier HW & SW for Normolization ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/RMSNorm make -C ${SH_DIR} sh-old-hs

norm-sw:
	@echo "Compiling SoftHier HW & SW for Normolization ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/RMSNorm make -C ${SH_DIR} sh-old-sw

norm-pre: norm-clean norm-cfg norm-pld norm-hs

norm-run: norm-pre
	@echo "SoftHier simulation for Normolization ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/RMSNorm pld=${CUR_DIR}/sw/RMSNorm/preload.elf make -C ${SH_DIR} sh-old-run

norm-run%: norm-pre
	@echo "SoftHier simulation for Normolization ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/RMSNorm pld=${CUR_DIR}/sw/RMSNorm/preload.elf make -C ${SH_DIR} sh-old-run$*

norm-clean:
	@echo "clean up intermidiate files generated for Normolization ..."
	rm -f ${CUR_DIR}/sw/RMSNorm/include/norm.h ${CUR_DIR}/sw/RMSNorm/preload.elf ${CUR_DIR}/sw/RMSNorm/preload.dump ${CUR_DIR}/sw/RMSNorm/include/preload.h ${SH_DIR}/dump_*



###################
#   Activation    #
###################

ACTI_FILE ?= ${CUR_DIR}/config/kernels/acti.py
ifdef acti
	ACTI_FILE = $(realpath $(acti))
endif

acti-cfg:
	@echo "Generating configuration files for Activation ..."
	python3 ${CUR_DIR}/scripts/kernels/acti_config.py ${ACTI_FILE} ${CUR_DIR}/sw/Activation/include/acti.h

acti-pld:
	@echo "Generating preload files for Activation ..."
	python3 ${CUR_DIR}/scripts/kernels/acti_preload.py \
			${CUR_DIR}/sw/Activation/preload.elf \
			${CUR_DIR}/sw/Activation/include/preload.h \
			${ARCH_FILE} \
			${ACTI_FILE}

acti-hs:
	@echo "Compiling SoftHier HW & SW for Activation ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/Activation make -C ${SH_DIR} sh-old-hs

acti-sw:
	@echo "Compiling SoftHier HW & SW for Activation ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/Activation make -C ${SH_DIR} sh-old-sw

acti-pre: acti-clean acti-cfg acti-pld acti-hs

acti-run: acti-pre
	@echo "SoftHier simulation for Activation ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/Activation pld=${CUR_DIR}/sw/Activation/preload.elf make -C ${SH_DIR} sh-old-run

acti-run%: acti-pre
	@echo "SoftHier simulation for Activation ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/sw/Activation pld=${CUR_DIR}/sw/Activation/preload.elf make -C ${SH_DIR} sh-old-run$*

acti-clean:
	@echo "clean up intermidiate files generated for Activation ..."
	rm -f ${CUR_DIR}/sw/Activation/include/acti.h ${CUR_DIR}/sw/Activation/preload.elf ${CUR_DIR}/sw/Activation/preload.dump ${CUR_DIR}/sw/Activation/include/preload.h ${SH_DIR}/dump_*

# Keep the existing default target and kernel flows unchanged.
include ${CUR_DIR}/scripts/sparse_dma/sparse_dma.mk
