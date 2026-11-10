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
  uint64_t vit_cycles      cycles the program measured around the model

It checks that each run halted, returned 0 and printed nothing to stderr,
that both programs produced the same output in every mode, and prints the
cycles side by side.

Usage: python3 run_vit.py
Exit code 0 if the outputs match everywhere, 1 otherwise.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CRUDA = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CRUDA)
import run_nn_tests as nt  # noqa: E402

ref = nt.ref
MODES = [("scalar", 0), ("superscalar", 2), ("superscalar", 4), ("superscalar", 10)]
BUDGET = 50_000_000
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


def run_program(name, source, kernels, tiled):
    out, binf, syms = build(name, source, kernels)
    rows = read_u32(binf, syms["vit_out_rows"][0])
    cols = read_u32(binf, syms["vit_out_cols"][0])
    want = (ref.pad16(rows) * ref.pad16(cols)) if tiled else rows * cols
    if syms["vit_out"][1] < want:
        raise SystemExit(f"{name}: vit_out is {syms['vit_out'][1]} bytes, "
                         f"{rows} x {cols} needs {want}")
    # One dump covering both vit_out and vit_cycles.
    lo = min(syms["vit_out"][0], syms["vit_cycles"][0])
    hi = max(syms["vit_out"][0] + want, syms["vit_cycles"][0] + 8)

    runs = {}
    for mode, width in MODES:
        tag = "scalar" if mode == "scalar" else f"w{width}"
        a0, total, dump, stderr = nt.simulate(binf, BUDGET, lo, hi - lo,
                                              os.path.join(out, f"dump_{tag}.txt"), mode, width)
        c = syms["vit_cycles"][0] - lo
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
                         model=int.from_bytes(dump[c:c + 8], "little"),
                         total=total, problems=problems)
    return runs


def main():
    nt.run(["make", "build/run_test"], cwd=os.path.join(nt.RSVP, "32I"))
    cruda = run_program("cruda", os.path.join(HERE, "vit.c"), kernel_files(), tiled=True)
    base = run_program("baseline", os.path.join(HERE, "vit_baseline.c"), [], tiled=False)

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
    print(f"output {shape[0]} x {shape[1]}\n")
    print(f"{'mode':<8} {'baseline model':>14} {'CRUDA model':>12} {'speedup':>8}"
          f"   {'baseline prog':>13} {'CRUDA prog':>11}")
    for tag in cruda:
        bm, cm = base[tag]["model"], cruda[tag]["model"]
        print(f"{tag:<8} {bm:>14,} {cm:>12,} {bm / max(cm, 1):>7.1f}x"
              f"   {base[tag]['total']:>13,} {cruda[tag]['total']:>11,}")
    print("\nOUTPUTS MATCH" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
