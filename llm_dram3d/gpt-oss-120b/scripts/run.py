#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Run the complete layer and require intermediate, output, route and KV checks."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess

STAGES = ("norm1", "qkv", "rope_kv_append", "attention", "out_projection",
          "attention_residual", "norm2", "router_projection", "topk_softmax",
          "experts_swiglu", "expert_combine_residual")


def report(text, manifest):
    match = re.search(r"LLM_DRAM3D_PASS batch=(\d+) seq=(\d+) past=(\d+) layer=(\d+) "
                      r"cycles_hi=(\d+) cycles_lo=(\d+) errors=0 checked=(\d+)", text)
    if not match or "MISMATCH" in text or "LLM_ROUTING_FAIL" in text:
        raise ValueError("Layer validation failed or missing")
    b, s, p, layer, hi, lo, checked = map(int, match.groups())
    cycles = (hi << 32) | lo
    if [b, s, p, layer] != [manifest[k] for k in ("batch", "sequence", "past_kv", "layer")]:
        raise ValueError("Unexpected layer/scenario in validation report")
    if checked != manifest["verified_elements"] or cycles <= 0:
        raise ValueError("Incomplete validation report")
    stages = re.findall(r"LLM_STAGE id=(\d+) cycles=(\d+)", text)
    if [int(i) for i, _ in stages] != list(range(len(STAGES))):
        raise ValueError("Missing stage measurements")
    stage_cycles = {STAGES[int(i)]: int(c) for i, c in stages}
    if sum(stage_cycles.values()) != cycles:
        raise ValueError("Inconsistent stage cycle totals")
    return dict(manifest, status="PASS", cycles=cycles, stage_cycles=stage_cycles,
                measured_clock_hz=1_000_000_000, layer_time_us=cycles / 1000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gvsoc", default="gvsoc")
    p.add_argument("--arch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    out = args.output.resolve()
    (out / "results.json").unlink(missing_ok=True)
    manifest = json.loads((out / "manifest.json").read_text())
    if args.arch.resolve() != Path(manifest["arch"]):
        raise SystemExit("Run with the architecture used for generation")
    env = dict(os.environ, SOFTHIER_ARCH_FILE=str(args.arch.resolve()))
    command = [args.gvsoc, "--target=pulp.chips.soft_hier_old.flex_cluster",
               f"--binary={out / 'layer.elf'}", f"--preload={out / 'preload.elf'}", "run"]
    logpath = out / "simulation.log"
    with logpath.open("w") as log:
        process = subprocess.Popen(command, cwd=out, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit(f"Simulation interrupted/timed out: {logpath}")
    if code:
        raise SystemExit(f"Simulator failed ({code}): {logpath}")
    try:
        result = report(logpath.read_text(), manifest)
    except ValueError as error:
        raise SystemExit(f"{error}: {logpath}") from error
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['case']}: PASS, {result['verified_elements']} checks, "
          f"{result['cycles']} cycles ({result['layer_time_us']:.3f} us)")


if __name__ == "__main__":
    main()
