# SPDX-License-Identifier: Apache-2.0
class SparseAttnAccess:
    def __init__(self):
        self.token_dtype = "fp16"
        self.token_dim = 128
        self.selected_tokens = 1024
        self.context_tokens = 131072
        self.selection_seed = 42
        self.sorted_indices = True
