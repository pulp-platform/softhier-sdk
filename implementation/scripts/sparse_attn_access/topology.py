# SPDX-License-Identifier: Apache-2.0
"""HBM enumeration used by the SoftHier chip's address map and DRAM instances."""


def hbm_layout(arch):
    nx, ny = arch['num_cluster_x'], arch['num_cluster_y']
    counts = arch['hbm_chan_placement']
    if (counts not in ([ny, 0, 0, 0], [ny, nx, ny, nx])
            or arch['num_node_per_ctrl'] != 1 or arch['hbm_node_aliase'] != 1):
        raise ValueError('Use one unaliased HBM channel per edge node: west-only or all four edges')
    nodes, slot = [], 0
    for edge, count, capacity in zip(('west', 'north', 'east', 'south'), counts, (ny, nx, ny, nx)):
        for index in range(count):
            position = {'west': [0,index+1], 'north': [index+1,ny+1],
                        'east': [nx+1,index+1], 'south': [index+1,0]}[edge]
            nodes.append({'id':len(nodes), 'edge':edge, 'edge_index':index,
                          'label':edge[0].upper()+str(index), 'position':position,
                          'base':arch['hbm_start_base']+(slot+index)*arch['hbm_node_addr_space'],
                          'channel':f'{edge}_hbm_chan_{index}', 'controller':f'{edge}_hbm_ctrl_{index}'})
        slot += capacity
    return nodes


def hbm_description(arch):
    nodes = hbm_layout(arch)
    return f"{len(nodes)} HBM nodes on " + ('the west edge' if arch['hbm_chan_placement'][1:] == [0,0,0] else 'all four edges')
