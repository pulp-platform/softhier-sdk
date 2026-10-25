#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Run the complete layer and record timing and optional numerical validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

STAGES = ("norm1", "qkv", "rope_kv_append", "attention", "out_projection",
          "attention_residual", "norm2", "router_projection", "topk_softmax",
          "experts_swiglu", "expert_combine_residual")


def report(text, manifest):
    matches = re.findall(r"LLM_DRAM3D_(PASS|DONE) batch=(\d+) seq=(\d+) past=(\d+) layer=(\d+) "
                         r"cycles_hi=(\d+) cycles_lo=(\d+) errors=0 checked=(\d+)", text)
    if (len(matches) != 1 or "MISMATCH" in text or "LLM_ROUTING_FAIL" in text
            or "LLM_DRAM3D_FAIL" in text):
        raise ValueError("Layer failed or completion report missing")
    status, *values = matches[0]
    b, s, p, layer, hi, lo, checked = map(int, values)
    cycles = (hi << 32) | lo
    if [b, s, p, layer] != [manifest[k] for k in ("batch", "sequence", "past_kv", "layer")]:
        raise ValueError("Unexpected layer/scenario in completion report")
    expected = manifest["verified_elements"] if status == "PASS" else 0
    if checked != expected or cycles <= 0:
        raise ValueError("Incomplete layer report")
    stages = re.findall(r"LLM_STAGE id=(\d+) cycles=(\d+)", text)
    if [int(i) for i, _ in stages] != list(range(len(STAGES))):
        raise ValueError("Missing stage measurements")
    stage_cycles = {STAGES[int(i)]: int(c) for i, c in stages}
    if sum(stage_cycles.values()) != cycles:
        raise ValueError("Inconsistent stage cycle totals")
    steps = re.findall(r"^LLM_OPT_STEP id=(\d+)$", text, re.M)
    if len(steps) > 1:
        raise ValueError("Ambiguous optimization step")
    return dict(manifest, status=status, verified_elements=checked,
                optimization_step=int(steps[0]) if steps else 0,
                cycles=cycles, stage_cycles=stage_cycles,
                measured_clock_hz=1_000_000_000, layer_time_us=cycles / 1000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gvsoc", default="gvsoc")
    p.add_argument("--arch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path)
    p.add_argument("--run-dir", type=Path, help="separate logs/results; reuse --output's generated data")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    out = args.output.resolve()
    run = args.run_dir.resolve() if args.run_dir else out
    run.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve() if args.binary else out / "layer.elf"
    (run / "results.json").unlink(missing_ok=True)
    manifest = json.loads((out / "manifest.json").read_text())
    if args.arch.resolve() != Path(manifest["arch"]):
        raise SystemExit("Run with the architecture used for generation")
    env = dict(os.environ, SOFTHIER_ARCH_FILE=str(args.arch.resolve()))
    command = [args.gvsoc, "--target=pulp.chips.soft_hier_old.flex_cluster",
               f"--binary={binary}", f"--preload={out / 'preload.elf'}", "run"]
    logpath = run / "simulation.log"
    started = time.monotonic()
    with logpath.open("w") as log:
        process = subprocess.Popen(command, cwd=run, env=env, stdout=log,
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
    result.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  command=command, host_seconds=time.monotonic() - started)
    (run / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['case']}: {result['status']}, {result['verified_elements']} checks, "
          f"{result['cycles']} cycles ({result['layer_time_us']:.3f} us)")


if __name__ == "__main__":
    main()
