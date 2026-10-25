#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Compile, validate and retain numbered software steps against one preload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arch", type=Path, required=True)
    p.add_argument("--gvsoc", default="gvsoc")
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(s < 0 or s > 26 for s in args.steps):
        p.error("steps must be distinct integers in 0..26")
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if Path(manifest["arch"]) != args.arch.resolve():
        p.error("use the architecture recorded in the generated manifest")
    inputs = {name: digest(out / name) for name in
              ("preload.elf", "manifest.json", "layer.h", "flex_cluster_arch.h", "memory.ld")}
    inputs[str(args.arch.resolve())] = digest(args.arch.resolve())
    sdk = APP.parents[1]
    compiler = sdk / "toolchain/install/bin/riscv32-unknown-elf-gcc"
    inputs[str(compiler)] = digest(compiler)
    for path in sorted((sdk / "runtime/runtime").rglob("*")):
        if path.is_file() and path.suffix in (".h", ".inc", ".s"):
            inputs[str(path)] = digest(path)
    for step in args.steps:
        run = out / "steps" / f"{step:02d}"
        run.mkdir(parents=True, exist_ok=False)
        source = run / "source"
        shutil.copytree(APP / "sw", source / "sw")
        shutil.copy2(APP / "Makefile", source / "Makefile")
        shutil.copytree(APP / "scripts", source / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        command = ["make", "-f", str(source / "Makefile"), "compile", f"SDK_DIR={APP.parents[1]}", f"OUT={out}",
                   f"ELF={run / 'layer.elf'}", f"OPT_STEP={step}", "VALIDATE=1"]
        evidence = dict(step=step, inputs_sha256=inputs, compile_command=command,
            source_sha256={str(path.relative_to(source)): digest(path)
                           for path in sorted(source.rglob("*")) if path.is_file()})
        (run / "build.json").write_text(json.dumps(evidence, indent=2) + "\n")
        with (run / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(source / "scripts/run.py"), "--output", str(out),
            "--arch", str(args.arch), "--gvsoc", args.gvsoc, "--binary", str(run / "layer.elf"),
            "--run-dir", str(run), "--timeout", str(args.timeout)], check=True)
        result = json.loads((run / "results.json").read_text())
        if result["optimization_step"] != step or result["status"] != "PASS":
            raise ValueError(f"step {step}: expected complete numerical validation")


if __name__ == "__main__":
    main()
