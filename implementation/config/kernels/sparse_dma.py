# SPDX-License-Identifier: Apache-2.0
class SparseDma:
    def __init__(self):
        # Match the original Sparse-DMA gather experiment.
        self.rows = 2048
        self.dim = 128
        self.selected = 128
        self.seed = 42
        # Reserve the low HBM addresses for the SDK runtime ELF.
        self.hbm_offset = 0x00100000
