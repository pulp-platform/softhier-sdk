#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one full-chip case and retain the traces used by the benchmark report."""
import hashlib
import json
import os
from collections import deque
from pathlib import Path
import re
import subprocess
import sys
import time

IMPL = Path(__file__).resolve().parents[2]
GVSOC = IMPL.parents[1]
ANSI = re.compile(r'\x1b\[[0-9;]*m')
KEEP = ('NOC_V2_', 'SAA_', 'Cluster Sync:', 'Injecting request to noc', 'Forwarding request to next router',
        'Sending request to target', 'Received request response', 'Received IO req',
        'Sending read burst to AXI', 'GATHER_ROW id=', 'INDEX_READ addr=', '[iDMA] Finished')


def main():
    output, variant = Path(sys.argv[1]).resolve(), sys.argv[2]
    manifest = json.loads((output/'workload.json').read_text())
    folder = output/variant
    preload = json.loads((folder/'preload.json').read_text())
    run = folder/'run'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'exit-status.txt').exists():
        raise SystemExit(f'Run already exists: {run}; remove this generated run directory before rerunning')
    command = ['timeout', os.environ.get('SAA_TIMEOUT', '900'), 'stdbuf', '-oL', '-eL', str(GVSOC/'install/bin/gvsoc'),
               '--target=pulp.chips.soft_hier_old.flex_cluster', '--core-model=fast',
               '--binary='+str(folder/'sw/softhier.elf'), '--preload='+str(folder/'preload.elf'), 'run',
               '--trace=/chip/cluster_.*/idma', '--trace=/chip/cluster_.*/cluster_registers',
               '--trace=/chip/data_noc', '--trace-level=trace']
    # GVSoC's legacy trace matcher uses basic POSIX expressions, so specify
    # each edge separately rather than relying on grouped alternation.
    command += ['--trace=/chip/'+edge+'_hbm_ctrl' for edge in ('west', 'north', 'east', 'south')]
    for part in preload.get('files', [])[1:]:
        command.append('--target-opt=chip/hbm_preloader/binary='+str(folder/part['name']))
    library = GVSOC/'build/sparse_dma/dramsys-rtl/build/lib/libDRAMSys_Simulator.so'
    (run/'invocation.json').write_text(json.dumps({'command': command,
        'data_noc_backend': os.environ.get('SOFTHIER_DATA_NOC', manifest['architecture'].get('data_noc_backend', 'legacy')),
        'workload_sha256': hashlib.sha256((output/'workload.json').read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256((folder/'sw/softhier.elf').read_bytes()).hexdigest(),
        'preload_sha256': hashlib.sha256((folder/'preload.elf').read_bytes()).hexdigest(),
        'preload_files': preload.get('files', []),
        'dram_library_sha256': hashlib.sha256(library.read_bytes()).hexdigest()}, indent=2)+'\n')
    print(f'Running sparse_attn_access / {variant}: {manifest["clusters"]} clusters', flush=True)
    started, active, results, errors, done = time.monotonic(), False, set(), [], False
    recent, last_progress, line_count = deque(maxlen=24), started, 0
    with (run/'simulation.log').open('w') as stream, (run/'startup.log').open('w') as startup:
        process = subprocess.Popen(command, cwd=run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for raw in process.stdout:
            line = ANSI.sub('', raw)
            recent.append(line.rstrip())
            line_count += 1
            now = time.monotonic()
            if now-last_progress >= 10:
                (run/'progress.json').write_text(json.dumps({'wall_seconds': now-started,
                    'trace_lines': line_count, 'recent': list(recent)}, indent=2)+'\n')
                last_progress = now
            if 'SAA_RUN_BEGIN' in line:
                active = True
            if not active and not re.match(r'^\d+:', line):
                startup.write(line)
                startup.flush()
            if 'SAA_' in line:
                print(line.rstrip(), flush=True)
            if 'SAA_RUN_DONE' in line:
                done = True
            if 'SAA_RESULT' in line:
                cid = re.search(r'cluster=(\d+)', line)
                err = re.search(r'errors=(\d+)', line)
                if not cid or not err or int(err[1]) or int(cid[1]) in results:
                    errors.append(line.strip())
                if cid:
                    results.add(int(cid[1]))
            failed = 'SAA_FAIL' in line or 'FATAL' in line or 'Traceback' in line or 'Segmentation fault' in line
            if failed:
                errors.append(line.strip())
                print(line.rstrip(), flush=True)
            if failed or 'SAA_' in line or (active and any(marker in line for marker in KEEP)):
                stream.write(line)
                if 'SAA_' in line:
                    stream.flush()
        status = process.wait()
    (run/'exit-status.txt').write_text(str(status)+'\n')
    (run/'runner.json').write_text(json.dumps({'wall_seconds': time.monotonic()-started,
        'exit_status': status, 'reported_clusters': sorted(results), 'errors': errors}, indent=2)+'\n')
    if status or errors or not done or results != set(range(manifest['clusters'])):
        (run/'failure-tail.log').write_text('\n'.join(recent)+'\n')
        raise SystemExit(f'SAA_RUN_FAIL variant={variant} status={status} clusters={len(results)}; see {run}')
    print(f'SAA_RUN_PASS variant={variant} clusters={len(results)}', flush=True)


if __name__ == '__main__':
    main()
