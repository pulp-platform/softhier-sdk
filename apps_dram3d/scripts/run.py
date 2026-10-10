#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Run the generated GEMM, require full validation, and record measured cycles."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess

from generate import load_arch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvsoc", default="gvsoc")
    parser.add_argument("--arch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--trace-redmule", action="store_true",
                        help="Record per-call engine timing and distinguish activity from useful MAC throughput")
    args = parser.parse_args()
    out = args.output.resolve()
    for name in ("results.json", "redmule_timing.json"):
        if (out / name).exists():
            (out / name).unlink()
    arch = load_arch(args.arch)
    env = dict(os.environ, SOFTHIER_ARCH_FILE=str(args.arch.resolve()))
    command = [args.gvsoc, "--target=pulp.chips.soft_hier_old.flex_cluster",
               f"--binary={out / 'gemm.elf'}", f"--preload={out / 'preload.elf'}"]
    if args.trace_redmule:
        command += ["--trace=redmule/trace", "--trace-level=debug"]
    command.append("run")
    with (out / "simulation.log").open("w") as log:
        process = subprocess.Popen(command, cwd=out, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit(f"Simulation interrupted or timed out: {out / 'simulation.log'}")
    text = (out / "simulation.log").read_text()
    match = re.search(r"DRAM3D_GEMM_PASS M=(\d+) N=(\d+) K=(\d+) cycles=(\d+) errors=0 checked=(\d+)", text)
    if code or not match or "DRAM3D_MISMATCH" in text:
        raise SystemExit(f"GEMM failed (exit {code}): {out / 'simulation.log'}")
    result = json.loads((out / "manifest.json").read_text())
    m, n, k, cycles, checked = map(int, match.groups())
    if [m, n, k] != result["shape"] or checked != m * n or cycles <= 0:
        raise SystemExit("Invalid validation report")
    result["cycles"] = cycles
    result["macs_per_cycle"] = m * n * k / cycles
    peak = arch.num_cluster_x * arch.num_cluster_y * arch.redmule_ce_height * arch.redmule_ce_width
    result["system_peak_fraction"] = result["macs_per_cycle"] / peak
    # Payload throughput over the timed kernel, excluding preload and checking.
    # The board clock is 1 GHz, so bytes/cycle also gives decimal GB/s.
    result["dram_payload_gb_per_s"] = result["kernel_dram_bytes"] / cycles
    if args.trace_redmule:
        trace = re.sub(r"\x1b\[[0-9;]*m", "", text)
        calls = [dict(cluster=int(cid), start_ns=int(start), end_ns=int(end), cycles=int(duration))
                 for cid, start, end, duration in re.findall(
                     r"/cluster_(\d+)/redmule/trace[^\n]*Finished : (\d+) ns ---> (\d+) ns"
                     r"[^\n]*\((\d+) cyc\)", trace)]
        clusters = arch.num_cluster_x * arch.num_cluster_y
        mt, nt, kt = result["tile"]
        expected_calls = (m // mt) * (n // nt) * (k // kt) // clusters
        for cid in range(clusters):
            if sum(call["cluster"] == cid for call in calls) != expected_calls:
                raise SystemExit(f"Incomplete RedMule trace for cluster {cid}")
        call_cycles = sum(call["cycles"] for call in calls)
        # Activity includes the model's internal preload/drain, from trigger
        # through its last TCDM store. This differs from useful MAC utilization.
        result["redmule_active_fraction"] = call_cycles / (clusters * cycles)
        result["redmule_call_efficiency"] = m * n * k / (
            call_cycles * arch.redmule_ce_height * arch.redmule_ce_width)
        (out / "redmule_timing.json").write_text(json.dumps(calls, indent=2) + "\n")
    result["status"] = "PASS"
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['case']} {result['dataflow']}: PASS, {checked} elements, "
          f"{cycles} cycles, {result['system_peak_fraction']:.1%} of system peak")


if __name__ == "__main__":
    main()
