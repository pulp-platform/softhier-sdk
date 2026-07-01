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

import os
import numpy as np
import preload as pld

if __name__ == '__main__':
    start_address = 0x00000000
    K = 512
    M = 512
    N = 512

    A_host = np.ones((M, K), dtype=np.float16)
    B_host = np.ones((K, N), dtype=np.float16)
    C_host = np.zeros((M, N), dtype=np.float16)
    golden = np.matmul(A_host, B_host).astype(np.float16)

    A_address = 64 + start_address
    B_address = 64 + A_host.nbytes + start_address
    C_address = 64 + A_host.nbytes + B_host.nbytes + start_address
    golden_address = 64 + A_host.nbytes + B_host.nbytes + C_host.nbytes + start_address
    # create a uint32 np array to store the addresses
    args = np.array([A_address, B_address, C_address, K, M, N, golden_address], dtype=np.uint32)
    # print args in hex

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pld.make_preload_elf(script_dir+"/preload.elf", [args, A_host, B_host, C_host, golden])
