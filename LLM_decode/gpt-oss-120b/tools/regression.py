#!/usr/bin/env python3
"""Check simulator FP16 conversion, vector tails/completion, and subword DMA."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--gvsoc', default=str(ROOT / 'install/bin/gvsoc'))
    args = parser.parse_args()
    out = args.build_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    native = out / 'fp16_model_test'
    subprocess.run([args.cxx, '-std=c++17', '-O2',
                    f'-I{ROOT / "pulp/pulp/chips/soft_hier_old"}',
                    str(HERE / 'tests/fp16_model.cpp'), '-o', str(native)], check=True)
    subprocess.run([str(native)], check=True)
    subprocess.run([sys.executable, str(HERE / 'tools/build.py'), '--build-dir', str(out),
                    '--source', str(HERE / 'tests/vector.c')], check=True)
    command = [args.gvsoc, '--target=pulp.chips.soft_hier_old.flex_cluster',
               f'--binary={out / "decode.elf"}', 'run']
    env = dict(os.environ, SOFTHIER_ARCH_FILE=str(HERE / 'config/arch.py'))
    log_path = out / 'simulation.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=out, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=120)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL); process.wait(); raise
    text = log_path.read_text()
    assert code == 0 and 'VECTOR_PASS' in text and 'VECTOR_FAIL' not in text, log_path
    print('FP16 vector, short FP32 exponential, and subword DMA checks PASS')


if __name__ == '__main__':
    main()
