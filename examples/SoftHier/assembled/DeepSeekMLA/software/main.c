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

#include "flex_runtime.h"
// #include "hello_world.h"
#include "MLA_decode_MHA.h"
#include <math.h>

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    MLA_Decode_MHA(
        hbm_west(0,0)/*MetaQ_base_address*/,
        hbm_south(0,0)/*CKVR_base_address*/,
        hbm_west(0,0)/*Output_base_address*/,
        8/*speculative_length*/,
        1024/*kv_sequence_length*/,
        8/*batch_size*/,
        2/*elem_size*/);

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}