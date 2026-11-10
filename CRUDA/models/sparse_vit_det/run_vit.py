#!/usr/bin/env python3
"""Builds and runs Sparse-ViT-Det's two RSVP programs and compares them:

  vit.c           -- the model on CRUDA's legos (NPU + GPU)
  vit_baseline.c  -- the same computation in plain C on the host CPU

Both programs are self-contained: each makes its own input and keeps its own
buffers. This script passes them nothing. It builds each one (vit.c linked
with every CRUDA kernel file, like a library), runs it on the emulator in
scalar mode and superscalar at widths 2/4/10, and reads back what the
program left in these globals:

  int8_t   vit_out[]       the output: tile layout in vit.c, row-major in
                           vit_baseline.c
  uint32_t vit_out_rows    its size
  uint32_t vit_out_cols
  uint64_t vit_cycles      cycles the program measured around the whole model
  uint64_t vit_cycles_<step>   ... and around each step (optional; both
                           programs must have the same steps, listed in the
                           order they appear in vit.c)

It checks that each run halted, returned 0 and printed nothing to stderr,
that both programs produced the same output in every mode, and prints the
cycles side by side.

Usage: python3 run_vit.py
Exit code 0 if the outputs match everywhere, 1 otherwise.
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CRUDA = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CRUDA)
import run_nn_tests as nt  # noqa: E402

ref = nt.ref
MODES = [("scalar", 0), ("superscalar", 2), ("superscalar", 4), ("superscalar", 10)]
BUDGET = 4_000_000_000     # cycles; the baseline's 8-batch GEMM is ~800M
OPT = "O2"                    # host code; GPU kernel files are always -O0


def kernel_files():
    """Every CRUDA kernel file with its -O level: files with SIMT code at -O0."""
    files = []
    for name in sorted(os.listdir(nt.KERNELS)):
        if name.endswith(".c"):
            path = os.path.join(nt.KERNELS, name)
            with open(path) as f:
                files.append((path, "O0" if "simt_isa.h" in f.read() else OPT))
    return files


def build(name, source, kernels):
    """Compiles start.s + source + kernels; returns (prog.bin, {symbol: (address, size)})."""
    out = os.path.join(nt.BUILD_DIR, "vit", name)
    os.makedirs(out, exist_ok=True)
    objs = []
    for i, (src, opt) in enumerate([(nt.START_S, OPT), (source, OPT)] + kernels):
        obj = os.path.join(out, f"{i}_{os.path.splitext(os.path.basename(src))[0]}.o")
        nt.run([nt.CC, *nt.CFLAGS, f"-{opt}", "-I", nt.INCLUDE, "-I", nt.SIMT_INC,
                "-I", nt.NPU_INC, "-c", src, "-o", obj])
        objs.append(obj)
    elf = os.path.join(out, "prog.elf")
    binf = os.path.join(out, "prog.bin")
    nt.run([nt.CC, *nt.CFLAGS, "-T", nt.LINK_LD, "-o", elf, *objs])
    nt.run([nt.OBJCOPY, "-O", "binary", elf, binf])
    syms = {}
    for line in nt.run([nt.NM, "-S", elf]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 4:
            syms[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
    for s in ("vit_out", "vit_out_rows", "vit_out_cols", "vit_cycles"):
        if s not in syms:
            raise SystemExit(f"{os.path.basename(source)} does not define the global {s}")
    return out, binf, syms


def read_u32(binf, addr):
    """A constant from the program image (linked at address 0)."""
    with open(binf, "rb") as f:
        f.seek(addr)
        return int.from_bytes(f.read(4), "little")


def step_names(source):
    """The vit_cycles_<step> globals, in the order they first appear in source."""
    with open(source) as f:
        names = re.findall(r"\bvit_cycles_(\w+)", f.read())
    return list(dict.fromkeys(names))


def run_program(name, source, kernels, tiled, steps):
    out, binf, syms = build(name, source, kernels)
    for st in steps:
        if f"vit_cycles_{st}" not in syms:
            raise SystemExit(f"{os.path.basename(source)} does not define the global vit_cycles_{st}")
    counters = ["vit_cycles"] + [f"vit_cycles_{st}" for st in steps]
    rows = read_u32(binf, syms["vit_out_rows"][0])
    cols = read_u32(binf, syms["vit_out_cols"][0])
    want = (ref.pad16(rows) * ref.pad16(cols)) if tiled else rows * cols
    if syms["vit_out"][1] < want:
        raise SystemExit(f"{name}: vit_out is {syms['vit_out'][1]} bytes, "
                         f"{rows} x {cols} needs {want}")
    # One dump covering vit_out and every cycle counter.
    lo = min([syms["vit_out"][0]] + [syms[k][0] for k in counters])
    hi = max([syms["vit_out"][0] + want] + [syms[k][0] + 8 for k in counters])

    def one_mode(mode, width):
        tag = "scalar" if mode == "scalar" else f"w{width}"
        return tag, nt.simulate(binf, BUDGET, lo, hi - lo,
                                os.path.join(out, f"dump_{tag}.txt"), mode, width)

    with ThreadPoolExecutor(len(MODES)) as pool:       # the modes run in parallel
        results = list(pool.map(lambda m: one_mode(*m), MODES))

    runs = {}
    for tag, (a0, total, dump, stderr) in results:
        cyc = {k: int.from_bytes(dump[syms[k][0] - lo:syms[k][0] - lo + 8], "little")
               for k in counters}
        o = syms["vit_out"][0] - lo
        y = dump[o:o + want]
        if tiled:
            y = ref.from_tiles(y, rows, cols)[1]
        else:
            y = np.frombuffer(y, dtype=np.int8).reshape(rows, cols)
        problems = []
        if total >= BUDGET:
            problems.append(f"did not halt within {BUDGET:,} cycles")
        if a0 != 0:
            problems.append(f"main returned {a0 - (1 << 32) if a0 >= 1 << 31 else a0}")
        if stderr.strip():
            problems.append("stderr: " + stderr.strip().splitlines()[0])
        runs[tag] = dict(out=np.asarray(y, dtype=np.int64),
                         steps=[cyc[f"vit_cycles_{st}"] for st in steps],
                         model=cyc["vit_cycles"], total=total, problems=problems)
    return runs


def main():
    nt.run(["make", "build/run_test"], cwd=os.path.join(nt.RSVP, "32I"))
    steps = step_names(os.path.join(HERE, "vit.c"))
    with ThreadPoolExecutor(2) as pool:                 # both programs at once
        fc = pool.submit(run_program, "cruda", os.path.join(HERE, "vit.c"),
                         kernel_files(), True, steps)
        fb = pool.submit(run_program, "baseline", os.path.join(HERE, "vit_baseline.c"),
                         [], False, steps)
        cruda, base = fc.result(), fb.result()

    ok = True
    for tag in cruda:
        for name, runs in (("vit.c", cruda), ("vit_baseline.c", base)):
            for p in runs[tag]["problems"]:
                ok = False
                print(f"FAIL  {name} [{tag}]: {p}")
        a, b = cruda[tag]["out"], base[tag]["out"]
        if a.shape != b.shape:
            ok = False
            print(f"FAIL  [{tag}]: output sizes differ: vit.c {a.shape}, vit_baseline.c {b.shape}")
        elif (a != b).any():
            ok = False
            bad = np.argwhere(a != b)
            r, c = bad[0]
            print(f"FAIL  [{tag}]: {len(bad)} of {a.size} outputs differ; first at "
                  f"({r}, {c}): vit.c {a[r, c]}, vit_baseline.c {b[r, c]}")

    shape = cruda["scalar"]["out"].shape
    print(f"output {shape[0]} x {shape[1]}; cycles:\n")
    print(f"{'mode':<7} {'step':<12} {'baseline':>13} {'CRUDA':>11} {'speedup':>8}")
    for tag in cruda:
        b, c = base[tag], cruda[tag]
        lines = list(zip(steps, b["steps"], c["steps"]))
        lines += [("model", b["model"], c["model"]), ("program", b["total"], c["total"])]
        for i, (st, bc, cc) in enumerate(lines):
            print(f"{tag if i == 0 else '':<7} {st:<12} {bc:>13,} {cc:>11,} {bc / max(cc, 1):>7.1f}x")
        print()
    print("OUTPUTS MATCH" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
