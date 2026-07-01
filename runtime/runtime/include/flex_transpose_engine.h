//
// Copyright (C) 2026 ETH Zurich and University of Bologna
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Chi Zhang <chizhang@ethz.ch>

#ifndef _FLEX_TRANSPOSE_ENGINE_H_
#define _FLEX_TRANSPOSE_ENGINE_H_

#include "flex_runtime.h"
#define TRANSPOSE_ENGINE_REG (ARCH_CLUSTER_REG_BASE + ARCH_CLUSTER_REG_SIZE + 64)

void flex_transpose_engine_config(uint32_t M, uint32_t N, uint32_t src_addr, uint32_t dst_addr, uint32_t elem_size)
{
	volatile uint32_t * m_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 0);
	volatile uint32_t * n_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 4);
	volatile uint32_t * x_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 8);
	volatile uint32_t * y_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 12);
	volatile uint32_t * e_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 16);

	*m_reg = M;
	*n_reg = N;
	*x_reg = src_addr;
	*y_reg = dst_addr;
	*e_reg = elem_size;
}

uint32_t flex_transpose_engine_trigger(){
	volatile uint32_t * start_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 20);
	return *start_reg;
}

uint32_t flex_transpose_engine_wait(){
	volatile uint32_t * wait_reg = (volatile uint32_t *) (TRANSPOSE_ENGINE_REG + 24);
	return *wait_reg;
}

#endif