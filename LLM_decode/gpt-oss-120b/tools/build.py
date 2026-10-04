#!/usr/bin/env python3
"""Build the single decoder ELF using SDK startup, barriers, DMA, and RedMule APIs."""
import argparse
import math
from pathlib import Path
import runpy
import shutil
import subprocess

HERE = Path(__file__).resolve().parents[1]
SDK = HERE.parents[1]


def arch_headers(out, arch_file):
    arch = runpy.run_path(str(arch_file))['FlexClusterArch']()
    include = out / 'runtime/include'
    shutil.copytree(SDK / 'runtime/runtime', out / 'runtime', dirs_exist_ok=True)
    c = ['#ifndef FLEXCLUSTERARCH_H', '#define FLEXCLUSTERARCH_H']
    s = []
    for k, v in vars(arch).items():
        name = 'ARCH_' + k.upper()
        if isinstance(v, int):
            c.append(f'#define {name} {hex(v)}')
            s.append(f'.set {name}, {hex(v)}')
        elif isinstance(v, list):
            c.append(f'#define {name} ' + '{' + ','.join(map(str, v)) + '}')
    cores = arch.spatz_attaced_core_list
    c += [f'#define ARCH_SPATZ_ATTACED_CORES {len(cores)}',
          '#define ARCH_SPATZ_ATTACED_CHECK_LIST {' + ','.join(str(int(i in cores)) for i in range(arch.num_core_per_cluster)) + '}',
          '#define ARCH_SPATZ_ATTACED_SID_LIST {' + ','.join(str(cores.index(i) if i in cores else 0) for i in range(arch.num_core_per_cluster)) + '}',
          '#endif']
    stack_shift = int(math.log2(arch.cluster_stack_size)) - (arch.num_core_per_cluster - 1).bit_length()
    s.append(f'.set ARCH_CLUSTER_STACK_OFFSET, {stack_shift}')
    (include / 'flex_cluster_arch.h').write_text('\n'.join(c) + '\n')
    (include / 'flex_cluster_arch.inc').write_text('\n'.join(s) + '\n')
    return arch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--cc', default='riscv32-unknown-elf-gcc')
    p.add_argument('--arch', type=Path, default=HERE / 'config/arch.py')
    p.add_argument('--source', type=Path, default=HERE / 'main.c')
    args = p.parse_args()
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arch = arch_headers(out, args.arch.resolve())
    # All C data is cluster-private. Model parameters and tensors live in the HBM preload.
    link = out / 'link.ld'
    link.write_text(f'''OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {{ IMEM (rwx) : ORIGIN = {arch.instruction_mem_base}, LENGTH = {arch.instruction_mem_size} }}
SECTIONS {{
 .text : {{ *(.init) *(.text*) *(.rodata*) . = ALIGN(8); }} > IMEM
 .data : {{ __global_pointer$ = . + 0x800; *(.sdata*) *(.data*) . = ALIGN(8); }} > IMEM
 .bss : {{ *(.sbss*) *(.bss*) *(COMMON) . = ALIGN(8); }} > IMEM
 __l1_heap_start = 0; __hbm_heap_start = {arch.hbm_start_base};
}}
''')
    command = [args.cc, '-march=rv32imafdv_zfh', '-mabi=ilp32d', '-mcmodel=medlow',
               '-O2', '-g', '-nostdlib', '-fno-builtin', '-fno-tree-vectorize',
               '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
               f'-I{out / "runtime/include"}', f'-I{HERE / "include"}', f'-I{out}',
               f'-T{link}', str(out / 'runtime/flex_start.s'), str(args.source.resolve()),
               '-o', str(out / 'decode.elf')]
    with (out / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(out / 'decode.elf')


if __name__ == '__main__':
    main()
