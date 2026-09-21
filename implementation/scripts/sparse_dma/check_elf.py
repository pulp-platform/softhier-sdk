#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the generated application against the selected real memory capacities."""
import json
from pathlib import Path
import sys
from elftools.elf.elffile import ELFFile

binary, manifest = map(Path, sys.argv[1:])
arch = json.loads(manifest.read_text())['architecture']
with binary.open('rb') as stream:
    elf = ELFFile(stream)
    for section in elf.iter_sections():
        if section['sh_size'] == 0 or not section['sh_flags'] & 2:
            continue
        base, end = section['sh_addr'], section['sh_addr']+section['sh_size']
        ranges = [(arch['cluster_tcdm_base'], arch['cluster_tcdm_size']),
                  (arch['instruction_mem_base'], arch['instruction_mem_size']),
                  (arch['hbm_start_base'], arch['hbm_node_addr_space'])]
        if not any(start <= base and end <= start+size for start,size in ranges):
            raise SystemExit(f'{section.name}: {base:#x}..{end:#x} exceeds architecture memory')
print('SPARSE_DMA_ELF_PASS', binary)
