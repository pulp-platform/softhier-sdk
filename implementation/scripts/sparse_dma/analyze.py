#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the full-chip DMA/NoC/HBM route, then plot cycles and HBM utilization."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from hbm_timing import audit_hbm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(sys.argv[1]).resolve()
IMPL = Path(__file__).resolve().parents[2]
GVSOC = IMPL.parents[1]
sys.path.insert(0, str(IMPL))
from scripts.noc_v2 import NoCTraceV2
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def require(condition, message):
    if not condition:
        raise SystemExit('SPARSE_DMA_ANALYSIS_FAIL: ' + message)


def read_variant(name, manifest):
    folder = OUTPUT / name
    log = ANSI.sub('', (folder / 'run/simulation.log').read_text())
    require((folder / 'run/exit-status.txt').read_text().strip() == '0', name + ': simulator failed')
    require('SPARSE_DMA_FAIL' not in log and 'SPARSE_DMA_MISMATCH' not in log, name + ': application failed')
    require(log.count('SPARSE_DMA_PASS variant=' + name) == 1, name + ': missing application PASS')
    records = re.findall(r'SPARSE_DMA_RESULT ([^\n]+)', log)
    require(len(records) == 1, name + ': missing or duplicate result')
    fields = dict(re.findall(r'(\w+)=([^ ]+)', records[0]))
    cfg = manifest['kernel']
    require(fields['variant'] == name and int(fields['errors']) == 0, name + ': invalid result')
    require(fields['checksum'] == manifest['expected_checksum'], name + ': checksum mismatch')
    for field, expected in [('rows', cfg['rows']), ('dim', cfg['dim']), ('selected', cfg['selected']),
                            ('row_bytes', manifest['row_bytes']), ('bytes', manifest['gather_bytes'])]:
        require(int(fields[field]) == expected, name + ': wrong ' + field)
    expected_descriptors = 1 if name == 'hw-gather' else cfg['selected']
    require(int(fields['transfers']) == expected_descriptors, name + ': wrong DMA descriptor count')
    require(int(fields['cycles']) > 0, name + ': invalid cycle interval')
    config_path = folder / 'run/gvsoc_config.json'
    config = json.loads(config_path.read_text())['target']
    chip, arch = config['chip'], manifest['architecture']
    clusters = [key for key in chip if re.fullmatch(r'cluster_\d+', key)]
    require(len(clusters) == arch['num_cluster_x'] * arch['num_cluster_y'], name + ': wrong cluster count')
    for key in clusters:
        cluster = chip[key]
        require(len([p for p in cluster if re.fullmatch(r'pe\d+', p)]) == arch['num_core_per_cluster'],
                name + ': wrong core count')
        dma = cluster['idma']
        require(dma['gather_enable'] is True and dma['transfer_queue_size'] == arch['idma_outstand_txn']
                and dma['burst_queue_size'] == arch['idma_outstand_burst'], name + ': wrong iDMA configuration')
        require(['idma->index', 'idma_index_router->input'] in cluster['bindings']
                and ['idma_index_router->tcdm', 'tcdm->dma_input'] in cluster['bindings'],
                name + ': index port is not connected to TCDM')
        interleaver = cluster['tcdm']['dma_interleaver']
        require(interleaver['nb_banks'] == arch['cluster_tcdm_bank_nb']
                and interleaver['bank_width'] * 8 == arch['cluster_tcdm_bank_width'],
                name + ': wrong TCDM bank geometry')
        require(cluster['idma_index_router']['mappings']['tcdm']['size'] == arch['cluster_tcdm_size'],
                name + ': wrong local TCDM range')
        require(cluster['instr_mem']['size'] == arch['instruction_mem_size'], name + ': wrong instruction memory size')
    require(chip['data_noc']['width'] * 8 == arch['noc_link_width'], name + ': wrong NoC width')
    hbm_channels = [key for key in chip if re.fullmatch(r'(west|north|east|south)_hbm_chan_\d+', key)]
    require(sorted(hbm_channels) == ['west_hbm_chan_' + str(i) for i in range(4)], name + ': wrong HBM placement')
    require(all(chip[key]['dram-type'] == arch['hbm_type'] for key in hbm_channels),
            name + ': wrong HBM configuration')
    matrix_base = manifest['matrix_addr']
    matrix_end = matrix_base + cfg['rows']*manifest['row_bytes']
    row_bytes = manifest['row_bytes']
    expected_rows = collections.Counter((matrix_base + i*row_bytes, row_bytes) for i in manifest['indices'])
    backend = chip.get('data_noc_backend', 'legacy')
    nodes = [dict(id=i, base=arch['hbm_start_base']+i*arch['hbm_node_addr_space'], position=[0,i+1]) for i in range(4)]
    v2 = NoCTraceV2(arch['num_cluster_x'], nodes, arch['hbm_node_addr_space'],
                     (matrix_base, matrix_end)) if backend == 'floonoc_v2' else None
    axi_reads, noc_beats, controllers = [], [], []
    for line in log.splitlines():
        if v2 and 'NOC_V2_' in line and re.match(r'^\d+:',line):
            v2.feed(line,int(line.split(':',1)[0]))
        if '/cluster_0/idma/axi_read/' in line and 'Sending read burst to AXI' in line:
            match = re.search(r'base: (0x[0-9a-f]+), size: (0x[0-9a-f]+)', line)
            addr, size = (int(x,16) for x in match.groups())
            if matrix_base <= addr < matrix_end:
                axi_reads.append((addr,size))
        if '/data_noc/ni_1_1/' in line and 'Injecting request to noc' in line:
            match = re.search(r'base: (0x[0-9a-f]+), size: (0x[0-9a-f]+), op_code: (\d+), destination: \((\d+), (\d+)\)', line)
            if match:
                addr, size = int(match[1],16), int(match[2],16)
                # Preloading writes use the same NoC before the first timed read.
                if axi_reads and matrix_base <= addr < matrix_end:
                    require(match.groups()[2:] == ('0','0','1'), name + ': wrong NoC operation or destination')
                    noc_beats.append((addr,size))
        if '/west_hbm_ctrl_' in line and 'Received IO req' in line:
            match = re.search(r'/west_hbm_ctrl_(\d+)/.*offset: (0x[0-9a-f]+), id: (\d+), size: (0x[0-9a-f]+), is_write: (\d+)', line)
            if match and match[5] == '0':
                addr = int(match[2],16) + manifest['architecture']['hbm_start_base']
                if matrix_base <= addr < matrix_end:
                    require(match[1] == '0' and match[3] == '0', name + ': wrong HBM controller or node')
                    controllers.append((addr,int(match[4],16)))
    require(collections.Counter(axi_reads) == expected_rows, name + ': row source/size mismatch on iDMA AXI')
    beat_bytes = manifest['architecture']['noc_link_width'] // 8
    expected_beats = collections.Counter()
    for addr,size in expected_rows.elements():
        while size:
            chunk = min(size, beat_bytes - addr % beat_bytes)
            expected_beats[(addr,chunk)] += 1
            addr += chunk
            size -= chunk
    if v2:
        v2.finish(collections.Counter({(0,addr,size):count for (addr,size),count in expected_rows.items()}))
        noc_beats = [(addr,size) for (_,addr,size) in v2.reads.elements()]
    else:
        require(collections.Counter(noc_beats) == expected_beats, name + ': NoC route/beat count mismatch')
    require(collections.Counter(controllers) == expected_beats, name + ': west HBM node 0 traffic mismatch')
    indices = re.findall(r'INDEX_READ addr=(0x[0-9a-f]+)', log)
    expected_index_reads = (cfg['selected']*2+7)//8 if name == 'hw-gather' else 0
    require(len(indices) == expected_index_reads, name + ': packed index read count mismatch')
    require(log.count('GATHER_ROW id=') == (cfg['selected'] if name == 'hw-gather' else 0),
            name + ': gather decomposition mismatch')
    timing = audit_hbm(folder / 'run', manifest, int(fields['cycles']), config['clock']['frequency'])
    return {'variant':name, 'data_noc_backend':backend, 'cycles':int(fields['cycles']), 'bytes':int(fields['bytes']),
            'dma_descriptors':expected_descriptors, 'axi_rows':len(axi_reads),
            'noc_beats':len(noc_beats), 'hbm_node0_reads':len(controllers),
            'index_word_reads':len(indices), 'checksum':fields['checksum'], 'errors':0,
            'clock_hz':config['clock']['frequency'],
            **{key:value for key,value in timing.items() if key != 'pseudo_channels'},
            'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'elf_sha256':hashlib.sha256((folder/'sw/softhier.elf').read_bytes()).hexdigest(),
            'log_sha256':hashlib.sha256((folder/'run/simulation.log').read_bytes()).hexdigest()}


def main():
    manifest = json.loads((OUTPUT/'workload.json').read_text())
    results = [read_variant(name,manifest) for name in ('core-loop','inlined-loop','hw-gather')]
    timing_valid = all(item['timing_valid'] for item in results)
    for item in results:
        item['speedup_vs_core_loop'] = results[0]['cycles']/item['cycles']
        item['hbm_utilization_percent'] = 100 * item['effective_gbps'] / item['hbm_peak_gbps']
    revisions = {name:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
                 for name,path in [('gvsoc',GVSOC),('pulp',GVSOC/'pulp'),('core',GVSOC/'core'),('sdk',IMPL.parent)]}
    sources = [p for p in IMPL.rglob('*') if p.is_file() and 'build' not in p.relative_to(IMPL).parts
               and p.suffix in ('.c','.h','.py','.sh','.mk')]
    sources += [GVSOC/'pulp/pulp/chips/soft_hier_old'/name for name in
                ('cluster_unit.py','flex_cluster.py','flex_cluster_arch.py')]
    sources += [GVSOC/'pulp/pulp/chips/soft_hier_old'/name for name in
                ('flex_mesh_noc_v2.py','noc_bridge.hpp','noc_bridge_legacy.cpp','noc_bridge_v2.cpp')]
    sources += [p for p in (GVSOC/'pulp/pulp/floonoc_v2').iterdir() if p.suffix in ('.cpp','.hpp','.py')]
    source_hashes = {str(p.relative_to(GVSOC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record = {'workload':manifest, 'results':results, 'revisions':revisions, 'source_sha256':source_hashes,
              'functional_valid':True, 'timing_valid':timing_valid,
              'scope':'Full SoftHier sanity test using SDK fast cores; no RTL accuracy claim.'}
    (OUTPUT/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    with (OUTPUT/'results.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax = plt.subplots(figsize=(7.2,4.3),layout='constrained')
    bars=ax.bar(['Core loop','Inlined loop','HW gather'],[r['cycles'] for r in results],
                color=['#74859c','#2e86ab','#16867c'],width=.6)
    ax.bar_label(bars,labels=[f"{r['cycles']:,}\n{r['speedup_vs_core_loop']:.2f}×" for r in results],padding=5)
    ax.set_ylim(0,max(r['cycles'] for r in results)*1.25)
    ax.set_ylabel('Core cycles (issue through DMA completion)')
    ax.set_title('SoftHier: cluster 0 → NoC → west HBM node 0\n128 selected rows × 256 bytes')
    ax.axhline(results[0]['minimum_data_cycles'], color='#b64036', linestyle='--', linewidth=1,
               label=f"HBM data-only floor: {results[0]['minimum_data_cycles']:.0f} cycles")
    ax.legend(loc='upper right', fontsize=9)
    if not timing_valid:
        fig.suptitle('TIMING INVALID: overlapping HBM bursts; raw simulator results', color='#b64036', fontsize=10)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    for suffix in ('png','svg','pdf'):
        fig.savefig(OUTPUT/f'cycles.{suffix}',dpi=180)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(7.2,4.8),layout='constrained')
    utilization = [r['hbm_utilization_percent'] for r in results]
    bars=ax.bar(['Core loop','Inlined loop','HW gather'],utilization,
                color=['#74859c','#2e86ab','#16867c'],width=.6)
    ax.bar_label(bars,labels=[f"{r['hbm_utilization_percent']:.2f}%\n{r['effective_gbps']:.2f} GB/s"
                             for r in results],padding=5)
    ax.set_ylim(0,max(100,max(utilization)*1.2))
    ax.set_ylabel('HBM node 0 bandwidth utilization (%)')
    ax.set_title('SoftHier: cluster 0 → NoC → west HBM node 0\n'
                 f"{manifest['kernel']['selected']} selected rows × {manifest['row_bytes']} bytes")
    fig.supxlabel(f"Utilization = read bytes / execution time / {results[0]['hbm_peak_gbps']:.3f} GB/s\n"
                  'Execution time includes DMA setup, issue and completion wait.',fontsize=9)
    if not timing_valid:
        fig.suptitle('TIMING INVALID: overlapping HBM bursts; raw simulator results', color='#b64036', fontsize=10)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    for suffix in ('png','svg','pdf'):
        fig.savefig(OUTPUT/f'hbm_bandwidth.{suffix}',dpi=180)
    plt.close(fig)
    arch, kernel = manifest['architecture'], manifest['kernel']
    rows=['# SoftHier sparse-DMA sanity test', '',
          'Path: cluster 0 → NoC → west HBM node 0 → local TCDM.',
          f"NoC backend: `{results[0]['data_noc_backend']}`.",
          f"HBM: `{arch['hbm_type']}`; chip clock: {results[0]['clock_hz'] // 1000000} MHz.",
          f"Workload: {kernel['selected']} rows × {manifest['row_bytes']} bytes, seed {kernel['seed']}.", '',
          '| Case | Cycles | Speedup | HBM GB/s | HBM utilization | DMA descriptors | HBM timing |',
          '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for item in results:
        rows.append(f"| {item['variant']} | {item['cycles']:,} | {item['speedup_vs_core_loop']:.2f}× | "
                    f"{item['effective_gbps']:.2f} | {item['hbm_utilization_percent']:.2f}% | "
                    f"{item['dma_descriptors']} | {'PASS' if item['timing_valid'] else 'FAIL'} |")
    if not timing_valid:
        rows += ['', '**HBM timing invalid: these counts cannot be used as performance results.**']
    rows += ['', '![Cycles](cycles.png)', '',
             '![HBM bandwidth utilization](hbm_bandwidth.png)', '',
             'HBM utilization is audited payload bytes divided by the measured execution time and',
             'the recorded peak of west HBM node 0. This kernel reads payload from one node;',
             'normalizing over all four enabled nodes would divide these percentages by four.', '',
             'All cases passed bit-exact output, index preload, guard and descriptor checks.',
             f"Traces verify 128 row reads, {results[0]['noc_beats']} NoC requests and 256 HBM-controller beats to west node 0, and 32 packed",
             'index reads for HW gather. HBM checks cover byte counts, burst overlap and bandwidth.',
             f"The recorded HBM peak is {results[0]['hbm_peak_gbps']:.2f} GB/s; "
             f"the data-only minimum is {results[0]['minimum_data_cycles']:.1f} cycles.", '',
             'Cycles include setup, issue and completion wait; preload and verification are excluded.',
             'The chip retains 4 × 4 clusters, five cores per cluster, 128 × 32-bit TCDM banks,',
             '64 DMA transactions, 256 burst slots and a 1024-bit NoC. Only cluster 0/core 0 runs.',
             'These are SoftHier sanity measurements, without an RTL cycle-accuracy claim.', '',
             '`results.csv` and `results.json` contain measurements and validation details.',
             'Binaries and traces are in each case directory. From `softhier-sdk/implementation`:', '',
             '```bash',
             f"make sparse-runv SOFTHIER_DATA_NOC={results[0]['data_noc_backend']} SPARSE_OUTPUT=\"$PWD/build/{OUTPUT.name}\"",
             f"make sparse-report SPARSE_OUTPUT=\"$PWD/build/{OUTPUT.name}\"",
             '```']
    (OUTPUT/'REPORT.md').write_text('\n'.join(rows)+'\n')
    print('SPARSE_DMA_FUNCTIONAL_PASS variants=3 path=cluster0-NoC-west-HBM-node0')
    print(OUTPUT/'REPORT.md')
    require(timing_valid, 'HBM timing audit failed; generated report and plots are marked TIMING INVALID')
    print('SPARSE_DMA_ANALYSIS_PASS variants=3 timing_valid=True')


if __name__ == '__main__':
    main()
