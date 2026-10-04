#!/usr/bin/env python3
"""Run the same decoder ELF and common weights for each requested batch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 64])
    p.add_argument('--gvsoc', default=str(ROOT / 'install/bin/gvsoc'))
    p.add_argument('--timeout', type=int, default=21600)
    args = p.parse_args()
    out = args.build_dir.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    env = dict(os.environ, SOFTHIER_ARCH_FILE=manifest['arch'])
    binary_sha = hashlib.sha256((out / 'decode.elf').read_bytes()).hexdigest()
    for batch in args.batches:
        directory = out / f'batch-{batch}'; directory.mkdir(exist_ok=True)
        inputs = out / f'input-b{batch}.elf'
        assert inputs.is_file(), inputs
        with inputs.open('rb') as source:
            input_sha = hashlib.file_digest(source, 'sha256').digest()
        case_sha = hashlib.sha256((out / 'manifest.json').read_bytes() +
                                  binary_sha.encode() + input_sha).hexdigest()
        # Independent batches may run concurrently. Duplicate requests wait for the
        # same case and reuse only the result produced while they were waiting.
        lock = (directory / '.run.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result_path = directory / 'result.json'
            if result_path.exists():
                completed = json.loads(result_path.read_text())
                if completed.get('case_sha256') == case_sha:
                    print(json.dumps(completed), flush=True)
                    lock.close()
                    continue
        cmd = [args.gvsoc, '--target=pulp.chips.soft_hier_old.flex_cluster',
               f'--binary={out / "decode.elf"}', f'--preload={out / "weights.elf"}',
               f'--config-opt=**/hbm_preloader/binary={inputs}',
               '--trace=loader', '--trace=ctrl_registers', 'run']
        (directory / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        start = time.monotonic()
        log_path = directory / 'simulation.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(cmd, cwd=directory, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise
        text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
        if code or 'DECODE_DONE' not in text or 'DECODE_FAIL' in text:
            raise RuntimeError(f'Simulation failed: {log_path}\n' + '\n'.join(text.splitlines()[-20:]))
        cycle_match = re.search(r'^LAYER_CYCLES (\d+)$', text, re.M)
        period = re.search(r'Execution period is (\d+) ns', text)
        routing = re.search(r'ROUTING distinct_experts (\d+) assignments (\d+)', text)
        result = dict(batch=batch, cached_tokens=manifest['model']['cached_tokens'],
                      cycles=int(cycle_match[1]), simulated_ns=int(period[1]),
                      stages={name: int(ticks) for name, ticks in re.findall(r'^STAGE (\w+) (\d+)$', text, re.M)},
                      distinct_experts=int(routing[1]), assignments=int(routing[2]),
                      host_seconds=time.monotonic() - start, binary_sha256=binary_sha,
                      case_sha256=case_sha)
        assert sum(result['stages'].values()) == result['cycles']
        assert 0 <= result['simulated_ns'] - result['cycles'] < 10000, 'Cycle-counter wrap or clock mismatch'
        assert result['assignments'] == batch * manifest['model']['experts_per_token']
        init = re.findall(r'Direct ELF preload complete .*cycle: (\d+)\)', text)
        assert len(init) == 17 and set(init) == {'0'}, 'Initialization advanced simulated time'
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        lock.close()


if __name__ == '__main__':
    main()
