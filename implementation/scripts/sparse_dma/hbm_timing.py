# SPDX-License-Identifier: Apache-2.0
"""Audit the actual HBM configuration and data-bus intervals recorded by DRAMSys."""
import hashlib
import json
from pathlib import Path
import sqlite3


def audit_hbm(run_folder, manifest, cycles, clock_hz):
    databases = sorted(Path(run_folder).glob('DRAMSysRecordable0_*_ch0.tdb'))
    if len(databases) != 1:
        raise ValueError('Expected exactly one west HBM node 0 DRAMSys database')
    database = databases[0]
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        unit, clock_ps, spec_json = connection.execute('SELECT UnitOfTime, clk, Memspec FROM GeneralInfo').fetchone()
        if unit != 'PS':
            raise ValueError('Expected DRAMSys timestamps in picoseconds')
        spec = json.loads(spec_json)['memspec']
        if spec['memoryType'] != 'HBM2':
            raise ValueError('Expected an HBM2-compatible memory specification')
        architecture, timing = spec['memarchitecturespec'], spec['memtimingspec']
        base = manifest['matrix_addr'] - manifest['architecture']['hbm_start_base']
        end = base + manifest['kernel']['rows'] * manifest['row_bytes']
        phases = connection.execute('''
            SELECT p.Rank, p.DataStrobeBegin, p.DataStrobeEnd, t.DataLength
            FROM Phases p JOIN Transactions t ON p.Transact = t.ID
            WHERE p.PhaseName IN ('RD', 'RDA') AND t.Command = 'R'
                  AND t.Address >= ? AND t.Address < ?
            ORDER BY p.DataStrobeBegin, p.ID
        ''', (base, end)).fetchall()
    finally:
        connection.close()
    total_bytes = sum(p[3] for p in phases)
    if total_bytes != manifest['gather_bytes']:
        raise ValueError('DRAMSys matrix-read byte count differs from the workload')
    # HBM2 represents the two independent pseudo-channel buses as ranks.
    per_pc = {}
    for pc in range(architecture['nbrOfPseudoChannels']):
        intervals = [(begin, finish) for rank, begin, finish, _ in phases if rank == pc]
        last_end, overlaps = 0, 0
        for begin, finish in intervals:
            overlaps += begin < last_end
            last_end = max(last_end, finish)
        per_pc[str(pc)] = {
            'read_bursts': len(intervals), 'overlapping_bursts': overlaps,
            'burst_duration_ps': sorted(set(finish - begin for begin, finish in intervals)),
            'minimum_start_gap_ps': min((b[0]-a[0] for a,b in zip(intervals, intervals[1:])), default=0),
        }
    data_rate = architecture['dataRate']
    burst_clocks = architecture['burstLength'] / data_rate
    # Use the recorded SystemC period, including rounding of HBM2E's 1800 MHz clock.
    tck = clock_ps * 1e-12
    peak_bps = (architecture['width'] / 8 * architecture['nbrOfDevices']
                * architecture['nbrOfPseudoChannels'] * data_rate / tck)
    minimum_cycles = total_bytes / peak_bps * clock_hz
    overlaps = sum(pc['overlapping_bursts'] for pc in per_pc.values())
    spacing_valid = min(timing['CCDS'], timing['CCDL']) >= burst_clocks
    return {
        'hbm_peak_gbps': peak_bps / 1e9,
        'minimum_data_cycles': minimum_cycles,
        'effective_gbps': total_bytes * clock_hz / cycles / 1e9,
        'hbm_read_bytes': total_bytes,
        'hbm_read_bursts': len(phases),
        'hbm_burst_ns': burst_clocks * tck * 1e9,
        'hbm_ccds_ns': timing['CCDS'] * tck * 1e9,
        'hbm_overlapping_bursts': overlaps,
        'hbm_spacing_valid': spacing_valid,
        'timing_valid': spacing_valid and overlaps == 0 and cycles >= minimum_cycles - 1e-9,
        'hbm_memspec_sha256': hashlib.sha256(spec_json.encode()).hexdigest(),
        'pseudo_channels': per_pc,
    }
