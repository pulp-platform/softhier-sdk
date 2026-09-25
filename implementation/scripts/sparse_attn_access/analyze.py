#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit full-chip sparse access, measure its execution window, and plot traffic."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np
from topology import hbm_description, hbm_layout

VARIANTS = ('core-loop', 'inlined-loop', 'hw-gather')
LABELS = ('Core loop', 'Inlined loop', 'HW gather')
COLORS = ('#74859c', '#2e86ab', '#16867c')
IMPL = Path(__file__).resolve().parents[2]
GVSOC = IMPL.parents[1]
sys.path.insert(0, str(IMPL))
from scripts.noc_v2 import NoCTraceV2, plot_traffic


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(address, size, width):
    while size:
        chunk = min(size, width-address % width)
        yield address, chunk
        address, size = address+chunk, size-chunk


def validate_config(config, manifest):
    chip, arch = config['chip'], manifest['architecture']
    clusters = [key for key in chip if re.fullmatch(r'cluster_\d+', key)]
    require(len(clusters) == manifest['clusters'], 'Wrong cluster count')
    for key in clusters:
        cluster, dma = chip[key], chip[key]['idma']
        require(len([key for key in cluster if re.fullmatch(r'pe\d+', key)]) == arch['num_core_per_cluster'], 'Wrong core count')
        require(dma['gather_enable'] and dma['transfer_queue_size'] == arch['idma_outstand_txn']
                and dma['burst_queue_size'] == arch['idma_outstand_burst'], 'Wrong DMA configuration')
        require(['idma->index', 'idma_index_router->input'] in cluster['bindings']
                and ['idma_index_router->tcdm', 'tcdm->dma_input'] in cluster['bindings'], 'Index port must connect to L1')
        require(cluster['idma_index_router']['mappings']['tcdm']['size'] == arch['cluster_tcdm_size'], 'Wrong L1 size')
        banks=cluster['tcdm']['dma_interleaver']
        require(banks['nb_banks'] == arch['cluster_tcdm_bank_nb'] and banks['bank_width']*8 == arch['cluster_tcdm_bank_width'], 'Wrong TCDM bank geometry')
    hbm = [key for key in chip if re.fullmatch(r'(west|east|north|south)_hbm_chan_\d+', key)]
    layout = hbm_layout(arch)
    require(len(layout) == manifest['hbm_nodes'], 'Wrong HBM node count')
    require(manifest.get('hbm_node_layout', layout) == layout, 'Inconsistent HBM node layout')
    require(set(hbm) == {node['channel'] for node in layout}, 'Wrong HBM placement')
    require(all(chip[key]['dram-type'] == arch['hbm_type'] for key in hbm), 'Wrong HBM type')
    require(chip['data_noc']['width']*8 == arch['noc_link_width'], 'Wrong NoC width')
    if chip.get('data_noc_backend') == 'floonoc_v2':
        noc = chip['data_noc']
        require(noc['ni_outstanding_reqs'] == arch['noc_outstanding']
                and noc['router_input_queue_size'] == arch.get('noc_router_input_queue_size', 2)
                and noc['narrow_width']*8 == arch.get('noc_narrow_link_width', 64), 'Wrong v2 queue/link configuration')
        require(all(item['remove_offset'] == item['base'] for item in noc['mappings'].values()),
                'V2 targets require base-address removal')
    return config['clock']['frequency']


def hbm_audit(folder, manifest, begin_ns, end_ns):
    nodes, total_bytes = [], 0
    duration_ns = end_ns-begin_ns
    arch = manifest['architecture']
    for info in hbm_layout(arch):
        node = info['id']
        databases = list(folder.glob(f'DRAMSysRecordable{node}_*_ch0.tdb'))
        require(len(databases) == 1, f'Missing HBM database for node {node}')
        db = sqlite3.connect(databases[0].resolve().as_uri()+'?mode=ro', uri=True)
        try:
            unit, clk, spec_json = db.execute('SELECT UnitOfTime, clk, Memspec FROM GeneralInfo').fetchone()
            require(unit == 'PS', 'HBM database must use picoseconds')
            spec = json.loads(spec_json)['memspec']
            memory, timing = spec['memarchitecturespec'], spec['memtimingspec']
            rows = db.execute('''SELECT p.Rank, p.DataStrobeBegin, p.DataStrobeEnd, t.Address, t.DataLength
                FROM Phases p JOIN Transactions t ON p.Transact = t.ID
                WHERE p.PhaseName IN ('RD','RDA') AND t.Command = 'R'
                ORDER BY p.DataStrobeBegin, p.ID''').fetchall()
        finally:
            db.close()
        burst_bytes = memory['width']//8 * memory['nbrOfDevices'] * memory['burstLength']
        expected = Counter()
        for user in manifest['users']:
            for token, mapped in zip(user['indices'], user['address_indices']):
                if token % manifest['hbm_nodes'] == node:
                    local = mapped*manifest['token_bytes']-info['base']
                    expected.update(split(local, manifest['token_bytes'], burst_bytes))
        # All data reads in the timed interval must belong to the requested workload.
        selected = [row for row in rows if row[1] >= begin_ns*1000 and row[2] <= end_ns*1000+1000]
        require(Counter((r[3],r[4]) for r in selected) == expected, f'HBM {node}: read address/byte mismatch in execution window')
        workload_rows = [row for row in rows if row[3] >= manifest['hbm_data_offset']]
        require(len(workload_rows) == len(selected), f'HBM {node}: payload read outside execution window')
        last = {}
        for rank, begin, end, _, _ in selected:
            require(begin >= last.get(rank, 0), f'HBM {node}: overlapping bursts on pseudo-channel {rank}')
            last[rank] = end
        burst_clocks = memory['burstLength']/memory['dataRate']
        require(min(timing['CCDS'], timing['CCDL']) >= burst_clocks, 'HBM burst spacing violates data-bus capacity')
        peak = memory['width']/8*memory['nbrOfDevices']*memory['nbrOfPseudoChannels']*memory['dataRate']/(clk/1000)
        byte_count = sum(r[4] for r in selected)
        utilization = byte_count/duration_ns/peak
        occupancy = sum(r[2]-r[1] for r in selected)/(duration_ns*1000*memory['nbrOfPseudoChannels'])
        require(utilization <= 1+1e-9 and abs(occupancy-utilization) < 1e-9, 'HBM bandwidth/data-strobe audit failed')
        nodes.append({'node': node, 'label': info['label'], 'edge': info['edge'],
                      'read_bytes': byte_count, 'read_bursts': len(selected),
                      'peak_gbps': peak, 'effective_gbps': byte_count/duration_ns,
                      'utilization_percent': utilization*100, 'data_bus_occupancy_percent': occupancy*100,
                      'clock_ps': clk, 'burst_bytes': burst_bytes, 'burst_duration_ps': clk*burst_clocks,
                      'overlapping_bursts': 0, 'memspec_sha256': hashlib.sha256(spec_json.encode()).hexdigest()})
        total_bytes += byte_count
    require(total_bytes == manifest['read_bytes'], 'HBM total bytes mismatch')
    return nodes


def read_case(output, variant, m):
    run = output/variant/'run'
    require((run/'exit-status.txt').read_text().strip() == '0', variant+': simulation failed')
    invocation=json.loads((run/'invocation.json').read_text())
    require(sha(output/variant/'sw/softhier.elf') == invocation['binary_sha256'], 'Program changed after simulation')
    require(sha(output/variant/'preload.elf') == invocation['preload_sha256'], 'Preload changed after simulation')
    for part in invocation.get('preload_files', []):
        require(sha(output/variant/part['name']) == part['sha256'], 'Preload part changed after simulation')
    if 'workload_sha256' in invocation:
        require(sha(output/'workload.json') == invocation['workload_sha256'], 'Workload changed after simulation')
    config = json.loads((run/'gvsoc_config.json').read_text())['target']
    clock_hz = validate_config(config, m)
    arch, cfg = m['architecture'], m['kernel']
    layout = hbm_layout(arch)
    controller_nodes = {node['controller']:node['id'] for node in layout}
    rows_expected = {u['cluster']: [(idx*m['token_bytes'],m['token_bytes']) for idx in u['address_indices']] for u in m['users']}
    beats_expected = Counter((cid,addr,size) for cid,rows in rows_expected.items()
                             for row in rows for addr,size in split(*row, arch['noc_link_width']//8))
    axi, noc, controllers, index_reads = Counter(), Counter(), Counter(), Counter()
    gathered, result, roi, inflight, edges = Counter(), {}, {}, {}, Counter()
    injections, responses = [], []
    matrix = np.zeros((m['clusters'],m['hbm_nodes']), dtype=np.int64)
    backend = config['chip'].get('data_noc_backend', 'legacy')
    require(invocation.get('data_noc_backend', backend) == backend, 'Wrong selected NoC backend')
    v2 = NoCTraceV2(arch['num_cluster_x'], layout, arch['hbm_node_addr_space']) if backend == 'floonoc_v2' else None
    done = False
    with (run/'simulation.log').open() as stream:
        for line in stream:
            require('SAA_FAIL' not in line and 'FATAL' not in line, variant+': failure in trace')
            if 'SAA_RUN_DONE' in line:
                done = True
            if 'SAA_RESULT' in line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                cid = int(fields['cluster'])
                require(cid not in result and fields['variant'] == variant and int(fields['errors']) == 0, 'Invalid software result')
                require(fields['checksum'] == m['users'][cid]['checksum'], 'Output checksum mismatch')
                descriptors = (cfg['selected_tokens']+m['batch_tokens']-1)//m['batch_tokens'] if variant == 'hw-gather' else cfg['selected_tokens']
                require(int(fields['transfers']) == descriptors, 'DMA completion count mismatch')
                result[cid] = {key:int(fields[key]) for key in ('start','end','cycles','transfers','errors')}
                result[cid]['checksum'] = fields['checksum']
            if 'Cluster Sync:' in line and f'Type = {m["roi_type"]}' in line:
                match = re.search(r'/cluster_(\d+)/.*Cluster Sync: (\d+) ns -> (\d+) ns', line)
                cid, begin, end = map(int, match.groups())
                require(cid not in roi and end > begin, 'Invalid or repeated execution annotation')
                roi[cid] = (begin,end)
            if not re.match(r'^\d+:', line):
                continue
            time_ps = int(line.split(':',1)[0])
            if v2 and 'NOC_V2_' in line:
                v2.feed(line, time_ps)
            if 'Sending read burst to AXI' in line:
                match = re.search(r'/cluster_(\d+)/.*base: (0x[0-9a-f]+), size: (0x[0-9a-f]+)', line)
                cid, address, size = int(match[1]), int(match[2],16), int(match[3],16)
                axi[cid,address,size] += 1
            if 'INDEX_READ addr=' in line:
                match = re.search(r'/cluster_(\d+)/.*INDEX_READ addr=(0x[0-9a-f]+)', line)
                index_reads[int(match[1]),int(match[2],16)] += 1
            if 'GATHER_ROW id=' in line:
                gathered[int(re.search(r'/cluster_(\d+)/',line)[1])] += 1
            if 'Injecting request to noc' in line:
                match = re.search(r'/ni_(\d+)_(\d+)/.*req: (0x[0-9a-f]+), base: (0x[0-9a-f]+), size: (0x[0-9a-f]+), op_code: (\d+), destination: \((\d+), (\d+)\)',line)
                x,y,ptr,address,size,op,dx,dy = match.groups()
                x,y,address,size,op,dx,dy = int(x),int(y),int(address,16),int(size,16),int(op),int(dx),int(dy)
                if address < arch['hbm_start_base'] or op != 0:
                    continue
                cid = (y-1)*arch['num_cluster_x']+x-1
                node = (address-arch['hbm_start_base'])//arch['hbm_node_addr_space']
                require(0 <= node < len(layout) and [dx,dy] == layout[node]['position']
                        and 0 <= cid < m['clusters'], 'Wrong HBM route')
                require(ptr not in inflight, 'NoC request reused before completion')
                inflight[ptr] = {'position':(x,y), 'destination':(dx,dy), 'delivered':False}
                noc[cid,address,size] += 1
                matrix[cid,node] += size
                injections.append(time_ps)
            if not v2 and ('Forwarding request to next router' in line or 'Sending request to target' in line):
                match = re.search(r'/router_(\d+)_(\d+)/.*req: (0x[0-9a-f]+), (?:next_position|position): \((\d+), (\d+)\)',line)
                x,y,ptr,dx,dy = match.groups()
                if ptr in inflight:
                    x,y,dx,dy = int(x),int(y),int(dx),int(dy)
                    state = inflight[ptr]
                    require(state['position'] == (x,y) and abs(dx-x)+abs(dy-y) == 1, 'Invalid NoC request path')
                    edges[x,y,dx,dy] += 1
                    state['position'] = (dx,dy)
                    if 'Sending request to target' in line:
                        require((dx,dy) == state['destination'], 'Wrong NoC destination')
                        state['delivered'] = True
            if 'Received request response' in line:
                ptr = re.search(r'req: (0x[0-9a-f]+)',line)[1]
                if ptr in inflight:
                    require(inflight.pop(ptr)['delivered'], 'NoC response before delivery')
                    responses.append(time_ps)
            if 'Received IO req' in line and '_hbm_ctrl_' in line:
                match = re.search(r'/((?:west|north|east|south)_hbm_ctrl_\d+)/.*offset: (0x[0-9a-f]+), id: (\d+), size: (0x[0-9a-f]+), is_write: (\d+)',line)
                if match and match[5] == '0':
                    require(match[1] in controller_nodes, 'Read from an inactive HBM edge')
                    node, address, mux, size = controller_nodes[match[1]], int(match[2],16), int(match[3]), int(match[4],16)
                    require(mux == 0, 'Unexpected HBM controller alias')
                    controllers[node,address,size] += 1
    require(done and set(result) == set(roi) == set(range(m['clusters'])), 'Missing cluster results or execution windows')
    if v2:
        v2.finish(Counter((cid,addr,size) for cid,rows in rows_expected.items() for addr,size in rows))
        noc, edges = v2.reads, v2.requests
        injections, responses = v2.request_times, v2.response_times
        for (cid, node), size in v2.returned.items():
            matrix[cid,node] = size
    else:
        require(not inflight and len(injections) == len(responses), 'NoC requests did not complete')
    require(axi == Counter((cid,addr,size) for cid,rows in rows_expected.items() for addr,size in rows), 'DMA row addresses/counts mismatch')
    if not v2:
        require(noc == beats_expected, 'NoC read addresses/counts mismatch')
    ctrl_expected = Counter()
    for (_,address,size), count in beats_expected.items():
        node = (address-arch['hbm_start_base'])//arch['hbm_node_addr_space']
        local = address-arch['hbm_start_base']-node*arch['hbm_node_addr_space']
        ctrl_expected[node,local,size] += count
    require(controllers == ctrl_expected, 'HBM controller traffic mismatch')
    preload = json.loads((output/variant/'preload.json').read_text())
    expected_indices = Counter((cid,preload['local_index_addr']+i*8)
                               for cid in range(m['clusters']) for i in range((cfg['selected_tokens']+1)//2)) if variant == 'hw-gather' else Counter()
    require(index_reads == expected_indices, 'Packed local index reads mismatch')
    require(gathered == (Counter({cid:cfg['selected_tokens'] for cid in range(m['clusters'])}) if variant == 'hw-gather' else Counter()), 'Gather decomposition mismatch')
    begin, end = min(r[0] for r in roi.values()), max(r[1] for r in roi.values())
    require(min(injections) >= begin*1000 and max(responses) <= end*1000, 'NoC requests outside measured execution window')
    for cid,(first,last) in roi.items():
        result[cid].update({'cluster':cid, 'roi_begin_ns':first, 'roi_end_ns':last})
        require(abs((last-first)*clock_hz/1e9-result[cid]['cycles']) <= 20, 'Cycle counter and global annotation disagree')
    nodes = hbm_audit(run,m,begin,end)
    peak = sum(n['peak_gbps'] for n in nodes)
    return {'variant':variant, 'data_noc_backend':backend,
            'noc_return_edges':[{'from':[x,y],'to':[dx,dy],'bytes':count} for (x,y,dx,dy),count in sorted(v2.data_bytes.items())] if v2 else [],
            'runtime_ns':end-begin, 'total_cycles':(end-begin)*clock_hz/1e9,
            'roi_begin_ns':begin, 'roi_end_ns':end, 'clock_hz':clock_hz, 'read_bytes':m['read_bytes'],
            'effective_gbps':m['read_bytes']/(end-begin), 'peak_gbps':peak,
            'hbm_utilization_percent':m['read_bytes']/(end-begin)/peak*100,
            'data_only_floor_ns':m['read_bytes']/peak, 'hbm_nodes':nodes,
            'clusters':[result[cid] for cid in sorted(result)], 'axi_rows':sum(axi.values()),
            'noc_requests':sum(noc.values()), 'index_word_reads':sum(index_reads.values()),
            'noc_edges':[{'from':[x,y],'to':[dx,dy],'requests':count} for (x,y,dx,dy),count in sorted(edges.items())],
            'cluster_hbm_bytes':matrix.tolist(), 'functional_valid':True, 'timing_valid':True,
            'config_sha256':sha(run/'gvsoc_config.json'), 'log_sha256':sha(run/'simulation.log'),
            'invocation':invocation,
            'simulator_wall_seconds':json.loads((run/'runner.json').read_text())['wall_seconds']}


def save(fig, output, name):
    for ext in ('png','svg','pdf'):
        fig.savefig(output/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def plots(output, results, manifest):
    layout=hbm_layout(manifest['architecture'])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    values = ([r['runtime_ns']/1000 for r in results], [r['hbm_utilization_percent'] for r in results])
    for ax,heights,ylabel,title in zip(axes,values,('Execution time (µs)','HBM bandwidth utilization (%)'),('Total runtime across all clusters',hbm_description(manifest['architecture']))):
        bars=ax.bar(LABELS,heights,color=COLORS,width=.62)
        ax.bar_label(bars,labels=[f'{value:.2f}' for value in heights],padding=5)
        ax.set_ylim(0,max(heights)*1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    axes[1].set_ylim(0,100)
    fig.suptitle(f"Sparse attention access · {manifest['clusters']} independent contexts · {manifest['kernel']['selected_tokens']:,} selected tokens each")
    save(fig,output,'runtime_hbm')
    fig,ax=plt.subplots(figsize=(max(8,.7*len(layout)),4.5),layout='constrained')
    x=np.arange(manifest['hbm_nodes'])
    for i,(result,label,color) in enumerate(zip(results,LABELS,COLORS)):
        bars=ax.bar(x+(i-1)*.24,[node['utilization_percent'] for node in result['hbm_nodes']],.24,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=7 if len(layout)>8 else 9,
                     rotation=90 if len(layout)>8 else 0)
    ax.set_xticks(x,[node['label'] for node in layout])
    ax.set_xlabel('HBM node: W = west, N = north, E = east, S = south')
    ax.set_ylabel('HBM utilization (%)')
    ax.set_title('Per-node utilization in the measured execution window')
    ax.set_ylim(0,100)
    ax.legend(ncol=3,loc='upper center')
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    save(fig,output,'hbm_nodes')
    if results[0]['data_noc_backend'] == 'floonoc_v2':
        plot_traffic(output, results, manifest, layout)
        return
    peak_rate=max(edge['requests']/(r['runtime_ns']/1000) for r in results for edge in r['noc_edges'])
    norm,cmap=Normalize(0,peak_rate),plt.get_cmap('YlOrRd')
    nx,ny=manifest['architecture']['num_cluster_x'],manifest['architecture']['num_cluster_y']
    for result,label in zip(results,LABELS):
        fig,(mesh,heat)=plt.subplots(1,2,figsize=(15,7),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
        for y in range(1,ny+1):
            for x in range(1,nx+1):
                if x < nx: mesh.plot([x,x+1],[y,y],color='#e1e5e8',zorder=0)
                if y < ny: mesh.plot([x,x],[y,y+1],color='#e1e5e8',zorder=0)
                mesh.scatter(x,y,s=370,color='#e5edf3',edgecolor='#60788c',zorder=3)
                mesh.text(x,y,f'C{(y-1)*nx+x-1}',ha='center',va='center',fontsize=9,zorder=4)
        for node in layout:
            x,y=node['position']
            mesh.scatter(x,y,s=470,marker='s',color='#d9eee8',edgecolor='#16867c',zorder=3)
            mesh.text(x,y,node['label'],ha='center',va='center',fontsize=9,zorder=4)
        for edge in result['noc_edges']:
            x,y=edge['from']; dx,dy=edge['to']
            rate=edge['requests']/(result['runtime_ns']/1000)
            ox,oy=(-(dy-y)*.06,(dx-x)*.06)
            mesh.add_patch(FancyArrowPatch((x+ox,y+oy),(dx+ox,dy+oy),arrowstyle='-|>',mutation_scale=11,
                                           linewidth=1+4*rate/peak_rate,color=cmap(norm(rate)),shrinkA=15,shrinkB=15,zorder=2))
        mesh.set(xlim=(-.65,nx+1.65),ylim=(-.65,ny+1.65),aspect='equal')
        mesh.set_xticks([]); mesh.set_yticks([])
        for spine in mesh.spines.values(): spine.set_visible(False)
        mesh.set_title('Measured read-request routes\nW/N/E/S = HBM edge · C = cluster')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=mesh,shrink=.72,label='Requests per µs on each directed link')
        im=heat.imshow(np.array(result['cluster_hbm_bytes'])/1024,aspect='auto',cmap='Blues',vmin=0,
                       vmax=max(max(row) for row in result['cluster_hbm_bytes'])/1024)
        heat.set_xticks(range(manifest['hbm_nodes']),[node['label'] for node in layout],
                        rotation=90 if len(layout)>8 else 0)
        for i in range(1,len(layout)):
            if layout[i]['edge'] != layout[i-1]['edge']:
                heat.axvline(i-.5,color='white',linewidth=1.5)
        step=max(1,manifest['clusters']//16)
        heat.set_yticks(range(0,manifest['clusters'],step),[f'C{i}' for i in range(0,manifest['clusters'],step)])
        heat.set_title('Read payload by source and destination')
        fig.colorbar(im,ax=heat,shrink=.72,label='KiB requested')
        fig.suptitle(f"{label} · {result['runtime_ns']/1000:.2f} µs · {result['effective_gbps']:.2f} GB/s")
        fig.supxlabel('Legacy NoC routes requests; read responses return by callback and have no separately simulated links.',fontsize=9)
        save(fig,output,'noc_'+result['variant'].replace('-','_'))


def compare_baseline(output, results, manifest, baseline_path):
    baseline_path=baseline_path.resolve()
    baseline=json.loads((baseline_path/'results.json').read_text())
    previous=baseline['workload']
    require(baseline['functional_valid'] and baseline['timing_valid'], 'Invalid baseline')
    require(previous['kernel'] == manifest['kernel'] and previous['clusters'] == manifest['clusters']
            and previous['batch_tokens'] == manifest['batch_tokens'], 'Baseline workload configuration differs')
    comparison_fields = {'hbm_chan_placement', 'data_noc_backend',
                         'noc_narrow_link_width', 'noc_router_input_queue_size'}
    require({k:v for k,v in previous['architecture'].items() if k not in comparison_fields} ==
            {k:v for k,v in manifest['architecture'].items() if k not in comparison_fields},
            'Baseline architecture differs beyond HBM placement or data NoC settings')
    for a,b in zip(previous['users'],manifest['users']):
        require(a['indices'] == b['indices'] and a['payload_sha256'] == b['payload_sha256'],
                'Baseline selections or token values differ')
    old={row['variant']:row for row in baseline['results']}
    rows=[]
    for current in results:
        reference=old[current['variant']]
        require(reference['invocation']['dram_library_sha256'] == current['invocation']['dram_library_sha256']
                and reference['hbm_nodes'][0]['memspec_sha256'] == current['hbm_nodes'][0]['memspec_sha256'],
                'Baseline HBM timing configuration differs')
        rows.append({'variant':current['variant'], 'baseline_hbm_nodes':previous['hbm_nodes'],
                     'current_hbm_nodes':manifest['hbm_nodes'],
                     'baseline_backend':reference.get('data_noc_backend', 'legacy'),
                     'current_backend':current['data_noc_backend'],
                     'baseline_cycles':reference['total_cycles'], 'current_cycles':current['total_cycles'],
                     'baseline_runtime_us':reference['runtime_ns']/1000, 'current_runtime_us':current['runtime_ns']/1000,
                     'speedup':reference['runtime_ns']/current['runtime_ns'],
                     'baseline_gbps':reference['effective_gbps'], 'current_gbps':current['effective_gbps'],
                     'baseline_utilization_percent':reference['hbm_utilization_percent'],
                     'current_utilization_percent':current['hbm_utilization_percent']})
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    x=np.arange(len(results))
    for ax,key,ylabel,title in zip(axes,('runtime_us','gbps','utilization_percent'),
                                   ('Execution time (µs)','Read bandwidth (GB/s)','HBM utilization (%)'),
                                   ('Total runtime','Aggregate read bandwidth','Utilization of enabled HBM')):
        for prefix,offset,color,label in [('baseline',-.18,'#74859c',f"{previous['hbm_nodes']} nodes / {old[results[0]['variant']].get('data_noc_backend', 'legacy')}"),
                                          ('current',.18,'#16867c',f"{manifest['hbm_nodes']} nodes / {results[0]['data_noc_backend']}")]:
            heights=[row[prefix+'_'+key] for row in rows]
            bars=ax.bar(x+offset,heights,.36,color=color,label=label)
            ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
        ax.set_xticks(x,LABELS)
        ax.set_ylabel(ylabel);ax.set_title(title)
        ax.set_ylim(0,100 if key == 'utilization_percent' else max(row[prefix+'_'+key] for row in rows for prefix in ('baseline','current'))*1.28)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(loc='upper right')
    fig.suptitle('Same token selections and payload · '
                 f"{previous['hbm_nodes']} HBM / {rows[0]['baseline_backend']} → "
                 f"{manifest['hbm_nodes']} HBM / {rows[0]['current_backend']}",fontsize=12)
    fig.supxlabel(f"Aggregate HBM peak: {baseline['results'][0]['peak_gbps']:.2f} → {results[0]['peak_gbps']:.2f} GB/s; utilization uses each configuration's own peak.",fontsize=10)
    save(fig,output,'comparison')
    return {'baseline_path':str(baseline_path), 'baseline_results_sha256':sha(baseline_path/'results.json'),
            'same_logical_indices_and_payload':True, 'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    manifest=json.loads((output/'workload.json').read_text())
    results=[read_case(output,variant,manifest) for variant in VARIANTS]
    for result in results:
        result['speedup_vs_core_loop']=results[0]['runtime_ns']/result['runtime_ns']
    revisions={name:{'head':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
                     'status':subprocess.check_output(['git','-C',str(path),'status','--short'],text=True).strip()}
               for name,path in [('gvsoc',GVSOC),('pulp',GVSOC/'pulp'),('core',GVSOC/'core'),('sdk',IMPL.parent)]}
    source_paths=list((IMPL/'sw/SparseAttnAccess').rglob('*'))+list((IMPL/'scripts/sparse_attn_access').rglob('*'))
    config_paths=manifest.get('configuration_files', {'architecture':str(IMPL/'config/arch/sparse_attn_access.py'),
                                                     'kernel':str(IMPL/'config/kernels/sparse_attn_access.py')})
    source_paths += [IMPL/'scripts/kernels/sparse_attn_access_preload.py',Path(config_paths['architecture']),
                     Path(config_paths['kernel']),GVSOC/'core/models/utils/loader/loader.cpp']
    source_paths += [IMPL/'scripts/noc_v2.py']
    source_paths += [GVSOC/'pulp/pulp/chips/soft_hier_old'/name for name in
                     ('flex_cluster.py','flex_mesh_noc_v2.py','noc_bridge.hpp','noc_bridge_legacy.cpp','noc_bridge_v2.cpp')]
    source_paths += [p for p in (GVSOC/'pulp/pulp/floonoc_v2').iterdir() if p.suffix in ('.cpp','.hpp','.py')]
    hashes={str(p.relative_to(GVSOC)):sha(p) for p in source_paths if p.is_file() and '__pycache__' not in p.parts}
    plots(output,results,manifest)
    comparison=compare_baseline(output,results,manifest,args.baseline) if args.baseline else None
    (output/'results.json').write_text(json.dumps({'workload':manifest,'results':results,'revisions':revisions,
        'source_sha256':hashes,'functional_valid':True,'timing_valid':True,'comparison':comparison},indent=2)+'\n')
    columns=['variant','data_noc_backend','total_cycles','runtime_ns','read_bytes','effective_gbps','peak_gbps','hbm_utilization_percent',
             'speedup_vs_core_loop','axi_rows','noc_requests','index_word_reads','simulator_wall_seconds']
    with (output/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    with (output/'clusters.csv').open('w',newline='') as stream:
        rows=[dict(variant=r['variant'],**c) for r in results for c in r['clusters']]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cfg,arch=manifest['kernel'],manifest['architecture']
    lines=['# Sparse attention access on SoftHier','',
        f"{arch['num_cluster_x']} × {arch['num_cluster_y']} clusters, one independent user context per cluster; {hbm_description(arch)}.",
        f"Each cluster selects {cfg['selected_tokens']:,} unique tokens out of {cfg['context_tokens']:,}, {cfg['token_dtype'].upper()} × {cfg['token_dim']}; seed {cfg['selection_seed']}, {'sorted' if cfg['sorted_indices'] else 'unsorted'} logical indices.",
        f"Read payload: {manifest['read_bytes']/2**20:g} MiB total, from {manifest['logical_context_bytes']/2**20:g} MiB of nonoverlapping context address space.",'',
        '| Case | Total cycles | Runtime (µs) | HBM GB/s | HBM utilization | Speedup |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['variant']} | {r['total_cycles']:,.0f} | {r['runtime_ns']/1000:.3f} | {r['effective_gbps']:.2f} | {r['hbm_utilization_percent']:.2f}% | {r['speedup_vs_core_loop']:.2f}× |")
    software_descriptors=cfg['selected_tokens']
    gather_descriptors=(cfg['selected_tokens']+manifest['batch_tokens']-1)//manifest['batch_tokens']
    lines += ['', f"HW gather reduces submissions from {software_descriptors:,} to {gather_descriptors:,} descriptors per cluster. "
        f"Its speedup is {results[2]['speedup_vs_core_loop']:.3f}× versus the core loop and "
        f"{results[1]['runtime_ns']/results[2]['runtime_ns']:.3f}× versus the inlined loop.",
        '', '![Runtime and HBM utilization](runtime_hbm.png)', '']
    if comparison:
        lines += ['## Comparison with the previous configuration', '',
                  'Logical token selections, token values, kernel settings and architecture parameters outside HBM placement and data NoC settings match the baseline. '
                  f"Data NoC: `{comparison['rows'][0]['baseline_backend']}` → `{comparison['rows'][0]['current_backend']}`.", '',
                  '| Case | Previous runtime (µs) | Current runtime (µs) | Speedup |',
                  '| --- | ---: | ---: | ---: |']
        for row in comparison['rows']:
            lines.append(f"| {row['variant']} | {row['baseline_runtime_us']:.3f} | {row['current_runtime_us']:.3f} | {row['speedup']:.2f}× |")
        lines += ['', '![Architecture comparison](comparison.png)', '',
                  'Utilization is normalized to the enabled HBM capacity in each configuration; read bandwidth in GB/s provides the absolute comparison.', '']
    lines += [
        f"NoC backend: `{results[0]['data_noc_backend']}`.", '',
        '## Measurement', '',
        'Runtime is the global span from the earliest cluster start annotation to the latest completion annotation. '
        'It includes DMA setup, issue, bounded-batch waits and final completion, and excludes preload, index generation, output initialization and verification. '
        'Per-cluster cycle counters are in `clusters.csv`; annotations add a few instructions around those counters.',
        f"Chip clock: {results[0]['clock_hz']/1e9:g} GHz; NoC link width: {arch['noc_link_width']} bits; HBM configuration: `{arch['hbm_type']}`.",
        f"HBM utilization = measured read bytes / runtime / aggregate peak. DRAMSys records {results[0]['hbm_nodes'][0]['clock_ps']} ps per HBM clock, "
        f"giving {results[0]['hbm_nodes'][0]['peak_gbps']:.3f} GB/s per node and {results[0]['peak_gbps']:.3f} GB/s total. "
        f"The aggregate data-only lower bound is {results[0]['data_only_floor_ns']/1000:.3f} µs.",
        '', '![HBM nodes](hbm_nodes.png)', '', '## Method and checks', '',
        '- One core per cluster issues transfers; the other cores participate in SDK barriers.',
        f"- All variants use identical selections, data, output order and {manifest['batch_tokens']}-token batches. "
        'The batch limit avoids exhaustion of the legacy private iDMA backend queues.',
        '- Core loop calls the SDK 1D DMA helper per token; inlined loop issues the same commands inline; HW gather consumes packed local 32-bit mapped indices.',
        '- Token t maps to HBM node t % H and local token t // H. Context stripes are densely adjacent per node after the reserved 1 MiB prefix. '
        'Mapped index = physical byte address / token bytes, with gather base zero and token-sized stride; logical sorted order is preserved across nodes.',
        '- HBM node enumeration follows SoftHier: west, north, east, south; increasing edge index within each edge.',
        '- Only selected HBM token values are initialized; unused context addresses retain their dense layout and are never accessed. All indices are preloaded directly into each cluster’s L1.',
        '- PASS: every selected element, per-cluster checksum, L1 guards, index hashes and DMA completion counts.',
        '- PASS: exact DMA, NoC and HBM read addresses/byte counts; local index reads; request paths and responses; HBM burst spacing and absence of same-bus overlap.',
        '', '## NoC traffic', '',
        'Arrows show measured read-request hop rates with a common scale across cases. Heatmaps show read payload by cluster and HBM node. ' +
        ('V2 plots include separately measured read-data return routes and payload bandwidth.' if results[0]['data_noc_backend'] == 'floonoc_v2' else
         'The legacy NoC returns data through response callbacks rather than independently routed response links; these plots do not measure return-link bandwidth.'), '']
    for variant,label in zip(VARIANTS,LABELS):
        lines += [f'![{label} NoC](noc_{variant.replace("-","_")}.png)', '']
    command=['make sparse_attn_access-run',
             'SPARSE_ATTN_ARCH_FILE="$PWD/'+str(Path(config_paths['architecture']).relative_to(IMPL))+'"',
             'SPARSE_ATTN_KERNEL_FILE="$PWD/'+str(Path(config_paths['kernel']).relative_to(IMPL))+'"',
             'SPARSE_ATTN_OUTPUT="$PWD/build/'+output.name+'"']
    if results[0]['data_noc_backend'] == 'floonoc_v2':
        command.append('SOFTHIER_DATA_NOC=floonoc_v2')
    if args.baseline:
        command.append('SPARSE_ATTN_BASELINE="'+str(args.baseline.resolve())+'"')
    lines += ['## Reproduction and scope', '',
        'From `softhier-sdk/implementation` (choose a fresh output directory when rerunning):', '',
        '```bash', (' \\'+'\n  ').join(command), '```', '',
        'Replace `sparse_attn_access-run` with `sparse_attn_access-report` to regenerate reports and PNG/SVG/PDF plots from saved results.',
        'The shared ELF loader resumes asynchronous completions on the next clock edge to avoid a zero-time SystemC stall during preload. '
        'This loader behavior is outside the measured kernel. The selected NoC backend and its bridges are included in the measured transfer time.',
        'These are full-chip GVSoC measurements with SDK fast cores and the HBM2E configuration inherited from sparse_dma; no RTL cycle-accuracy claim is made for this multi-cluster workload.',
        '`results.json` contains detailed audits and source/binary hashes; `results.csv` contains the case summary. Simulator wall time is recorded separately from simulated runtime.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('SAA_ANALYSIS_PASS variants=3 clusters='+str(manifest['clusters']))
    print(output/'REPORT.md')


if __name__ == '__main__':
    main()
