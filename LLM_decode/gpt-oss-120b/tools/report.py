#!/usr/bin/env python3
"""Summarize measured, numerically validated decoder runs."""
import argparse
import json
from pathlib import Path
import runpy

from reference import read_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 64])
    args = parser.parse_args()
    directory = args.build_dir.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    runs, validations = [], []
    for batch in args.batches:
        runs.append(json.loads((directory / f'batch-{batch}/result.json').read_text()))
        checks = json.loads((directory / f'batch-{batch}/validation.json').read_text())
        assert all(c['pass_check'] for c in checks.values()), batch
        validations.append(checks)
    assert len({r['binary_sha256'] for r in runs}) == 1, 'Batches used different programs'
    assert all(0 <= r['simulated_ns'] - r['cycles'] < 10000 for r in runs)
    arch = runpy.run_path(manifest['arch'])['FlexClusterArch']()
    address = manifest['tensors']['output']['address']
    previous = b''
    for r in sorted(runs, key=lambda r: r['batch']):
        blocks = read_dump(directory / f'batch-{r["batch"]}/dump_0', arch.hbm_start_base)
        data = bytearray()
        size = r['batch'] * manifest['model']['hidden_size'] * 2
        while len(data) < size:
            data += blocks[address + len(data)]
        assert len(data) == size and data[:len(previous)] == previous, 'Shared sequence output changed across batches'
        previous = data
    lines = ['# GPT-OSS-120B one-layer decode results', '',
             f'Full attention; {manifest["model"]["cached_tokens"]:,} cached tokens plus one new token. '
             'FP16 tensors, reproducible synthetic parameters, 1 GHz simulator clock.', '',
             'The timed region includes all 14 stages, DMA, and synchronization. '
             'Direct ELF initialization, output dumping, and reporting are outside it. '
             'Latency is for the complete batch; all sequences decode one token concurrently.', '',
             '| Batch | Cycles | Layer latency (ms) | Distinct experts | Output max absolute error |',
             '| ---: | ---: | ---: | ---: | ---: |']
    for r, check in zip(runs, validations):
        lines.append(f'| {r["batch"]} | {r["cycles"]:,} | {r["cycles"] / 1e6:.6f} | '
                     f'{r["distinct_experts"]} | {check["output"]["max_abs_error"]:.8g} |')
    lines += ['', 'Stage times in microseconds at 1 GHz:', '',
              '| Stage | ' + ' | '.join(f'Batch {r["batch"]}' for r in runs) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in runs) + ' |']
    for stage in runs[0]['stages']:
        lines.append(f'| {stage} | ' + ' | '.join(f'{r["stages"][stage] / 1000:.3f}' for r in runs) + ' |')
    lines += ['', f'Common decoder ELF SHA-256: `{runs[0]["binary_sha256"]}`.', '',
              'Shared sequences produce byte-identical simulator outputs across these batch sizes.', '',
              f'All {manifest["model"]["num_experts"]} experts are preloaded; '
              f'the packed weight payload is {manifest["weights_bytes"]:,} bytes. '
              'The synthetic router uses distinct, reproducible top-k choices. '
              'Latency depends on expert reuse and is not a prediction for a trained checkpoint.', '',
              'This baseline uses one controlling scalar/vector core and one RedMule per cluster '
              'across 16 clusters. It does not exploit all four vector cores per cluster. '
              'FP32 reductions and FP32 accelerator accumulation are narrowed to FP16 at tensor/tile boundaries.', '',
              'The adjacent manifest.json, batch-*/command.json, simulation.log, result.json, '
              'validation.json, and dump_0 contain the configuration, raw measurements, and numerical evidence.', '']
    path = directory / 'RESULTS.md'; path.write_text('\n'.join(lines))
    print(path)


if __name__ == '__main__':
    main()
