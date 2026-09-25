# SPDX-License-Identifier: Apache-2.0
"""Audit FlooNoC v2 flits and draw request/return-data traffic separately."""
from collections import Counter
import re


class NoCTraceV2:
    def __init__(self, nx, nodes, node_size, address_range=None):
        self.nx = nx
        self.address_range = address_range
        self.nodes = [dict(n, size=node_size) for n in nodes]
        self.by_position = {tuple(n['position']): n['id'] for n in nodes}
        self.pending = {}
        self.reads = Counter()
        self.returned = Counter()
        self.requests = Counter()
        self.data_bytes = Counter()
        self.request_times, self.response_times = [], []

    def feed(self, line, time_ps):
        if 'NOC_V2_INJECT' in line:
            match = re.search(r'req=(0x[0-9a-f]+) addr=(0x[0-9a-f]+) bytes=(\d+) src=\((\d+),(\d+)\) dst=\((\d+),(\d+)\) rsp=(\d) write=(\d) nw=(\d)', line)
            if not match:
                raise ValueError('Malformed v2 injection trace')
            ptr, addr, size, x, y, dx, dy, rsp, write, nw = match.groups()
            if int(write):
                return
            addr, size, x, y, dx, dy, rsp, nw = int(addr,16), *map(int,(size,x,y,dx,dy,rsp,nw))
            if self.address_range and not self.address_range[0] <= addr < self.address_range[1]:
                return
            source, dest = (x,y), (dx,dy)
            if rsp:
                node = self.by_position[source]
                cid = (dy-1)*self.nx+dx-1
                if nw != 2:
                    raise ValueError('Wide read data must traverse the wide network')
            else:
                node = self.by_position[dest]
                cid = (y-1)*self.nx+x-1
                if nw != 0 or not (self.nodes[node]['base'] <= addr < self.nodes[node]['base']+self.nodes[node]['size']):
                    raise ValueError('Wrong v2 request network or HBM destination')
                self.reads[cid,addr,size] += 1
                self.request_times.append(time_ps)
            if ptr in self.pending:
                raise ValueError('V2 flit reused before ejection')
            self.pending[ptr] = dict(position=source, dest=dest, size=size, rsp=rsp, cid=cid, node=node)
        elif 'NOC_V2_HOP' in line:
            match = re.search(r'req=(0x[0-9a-f]+) from=\((\d+),(\d+)\) to=\((\d+),(\d+)\)', line)
            ptr,x,y,dx,dy = match.groups()
            if ptr not in self.pending:
                return
            state = self.pending[ptr]
            current, dest = tuple(map(int,(x,y))), tuple(map(int,(dx,dy)))
            # Border NIs inject into the adjacent router. Its ingress link has
            # no separate router event, so include that physical edge here.
            if state['position'] != current:
                self.hop(state, current)
            if current != dest:
                self.hop(state, dest)
        elif 'NOC_V2_EJECT' in line:
            match = re.search(r'req=(0x[0-9a-f]+) at=\((\d+),(\d+)\)', line)
            ptr,x,y = match.groups()
            if ptr not in self.pending:
                return
            state = self.pending.pop(ptr)
            if state['position'] != tuple(map(int,(x,y))) or state['position'] != state['dest']:
                raise ValueError('V2 flit ejected at the wrong node')
            if state['rsp']:
                self.returned[state['cid'],state['node']] += state['size']
                self.response_times.append(time_ps)

    def hop(self, state, dest):
        x,y = state['position']; dx,dy = dest
        if abs(dx-x)+abs(dy-y) != 1:
            raise ValueError('Nonadjacent v2 mesh hop')
        if state['rsp']:
            self.data_bytes[x,y,dx,dy] += state['size']
        else:
            self.requests[x,y,dx,dy] += 1
        state['position'] = dest

    def finish(self, expected_reads):
        if self.pending or self.reads != expected_reads:
            raise ValueError('V2 outstanding flits or wrong read addresses/counts')
        expected = Counter()
        for (cid,addr,size),count in expected_reads.items():
            node = next(n['id'] for n in self.nodes if n['base'] <= addr < n['base']+n['size'])
            expected[cid,node] += size*count
        if self.returned != expected:
            raise ValueError('V2 returned payload byte counts differ from requests')


def plot_traffic(output, results, manifest, layout):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch
    import numpy as np
    nx,ny = manifest['architecture']['num_cluster_x'],manifest['architecture']['num_cluster_y']
    scales = [max(e['requests']/(r['runtime_ns']/1000) for r in results for e in r['noc_edges']),
              max(e['bytes']/r['runtime_ns'] for r in results for e in r['noc_return_edges'])]
    for result in results:
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
        for ax,edges,scale,key,title,unit in zip(axes[:2],
                (result['noc_edges'],result['noc_return_edges']),scales,('requests','bytes'),
                ('Read-request network','Read-data return network'),('Requests / µs','Payload GB/s')):
            norm,cmap=Normalize(0,scale),plt.get_cmap('YlOrRd')
            for y in range(1,ny+1):
                for x in range(1,nx+1):
                    if x<nx: ax.plot([x,x+1],[y,y],color='#e1e5e8',zorder=0)
                    if y<ny: ax.plot([x,x],[y,y+1],color='#e1e5e8',zorder=0)
                    ax.scatter(x,y,s=260,color='#e5edf3',edgecolor='#60788c',zorder=3)
                    ax.text(x,y,f'C{(y-1)*nx+x-1}',ha='center',va='center',fontsize=8,zorder=4)
            for node in layout:
                x,y=node['position']
                ax.scatter(x,y,s=310,marker='s',color='#d9eee8',edgecolor='#16867c',zorder=3)
                ax.text(x,y,node['label'],ha='center',va='center',fontsize=8,zorder=4)
            for edge in edges:
                x,y=edge['from']; dx,dy=edge['to']
                rate=edge[key]/result['runtime_ns']*(1000 if key=='requests' else 1)
                ox,oy=(-(dy-y)*.065,(dx-x)*.065)
                ax.add_patch(FancyArrowPatch((x+ox,y+oy),(dx+ox,dy+oy),arrowstyle='-|>',
                    mutation_scale=10,linewidth=1+3*rate/scale,color=cmap(norm(rate)),shrinkA=12,shrinkB=12))
            ax.set(xlim=(-.7,nx+1.7),ylim=(-.7,ny+1.7),aspect='equal',title=title,xticks=[],yticks=[])
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.7,label=unit)
        im=axes[2].imshow(np.array(result['cluster_hbm_bytes'])/1024,aspect='auto',cmap='Blues')
        axes[2].set_xticks(range(len(layout)),[n['label'] for n in layout],rotation=90)
        axes[2].set_yticks(range(manifest['clusters']),[f'C{i}' for i in range(manifest['clusters'])])
        axes[2].set_title('Payload by cluster and HBM node')
        fig.colorbar(im,ax=axes[2],shrink=.7,label='KiB')
        fig.suptitle(f"{result['variant']} · {result['runtime_ns']/1000:.3f} µs · {result['effective_gbps']:.2f} GB/s")
        fig.supxlabel('FlooNoC v2: measured request and read-data routes; preload traffic is excluded.',fontsize=9)
        for ext in ('png','svg','pdf'):
            fig.savefig(output/f"noc_{result['variant'].replace('-','_')}.{ext}",dpi=180)
        plt.close(fig)
