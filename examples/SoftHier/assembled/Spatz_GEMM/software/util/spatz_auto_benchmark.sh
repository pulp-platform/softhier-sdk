#!/bin/bash

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

# List of P values to test
# P_VALUES=(32 128 512 1024)
P_VALUES=(128)

for P in "${P_VALUES[@]}"; do
  echo "Running with P=$P"

  # Run data generation script
  python examples/SoftHier/assembled/Spatz_GEMM/software/util/spatz_matmul_datagen.py \
    -M 32 -N 32 -P $P -spN 1 -spM 2 --sparse --idx_compact

  # Run make commands
  cfg=examples/SoftHier/assembled/Spatz_GEMM/config/arch_spatz.py \
  app=examples/SoftHier/assembled/Spatz_GEMM/software \
  make hs; make run
done