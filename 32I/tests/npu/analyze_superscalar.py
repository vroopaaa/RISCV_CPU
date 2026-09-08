#!/usr/bin/env python3
"""
Superscalar issue-width analysis driver.

For a given bare-metal C source (default: build/matmul_256_t2.c, the 256x256
NPU matmul used to develop the superscalar model -- see ../../../docs/
superscalar.md), this script:
  1. Cross-compiles it once per optimization level (-O0..-O3 by default)
     using tests/python/start.s + link.ld, same as `emul`/`assemblyinstruction`.
  2. Builds scripts/lib/trace_harness.cpp once (shared across every run).
  3. For each optimization level, runs the tracer in scalar mode and in
     superscalar mode at issue widths 1..--max-n, all against the SAME
     compiled binary for that level.
  4. Parses each resulting trace for: total cycles, total instructions
     (invariant across n for a fixed binary -- verified, not assumed), the
     full distribution of how many instructions co-issued per cycle, average
     instructions-per-cycle (IPC), and speedup vs. that level's own scalar
     baseline.
  5. Prints a report, and (unless --no-keep-traces) saves every trace to
     build/<name>_<OPT>_<scalar|nN>.txt, matching the naming already used
     there.

Run from anywhere -- paths are resolved relative to this script's location.

Usage:
  python3 analyze_superscalar.py                          # defaults: build/matmul_256_t2.c, O0-O3, n=1..6
  python3 analyze_superscalar.py --opts O2 O3              # just some optimization levels
  python3 analyze_superscalar.py --max-n 8                 # wider issue-width sweep
  python3 analyze_superscalar.py --src build/other.c        # a different (already-generated) source
  python3 analyze_superscalar.py --json build/report.json   # also write the computed stats as JSON
  python3 analyze_superscalar.py --no-keep-traces            # don't leave the .txt traces behind

Exit code is 0 if every run halted normally, 1 if any run hit its cycle cap
(the program never halted -- results for that run are still reported, just
flagged, since a truncated trace is still informative for the distribution).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ISA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))       # tests/npu -> tests -> 32I
REPO_ROOT = os.path.abspath(os.path.join(ISA_DIR, ".."))              # 32I -> repo root
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

START_S = os.path.join(ISA_DIR, "tests", "python", "start.s")
LINK_LD = os.path.join(ISA_DIR, "tests", "python", "link.ld")
TRACE_HARNESS_SRC = os.path.join(REPO_ROOT, "scripts", "lib", "trace_harness.cpp")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_OBJDUMP = "riscv64-unknown-elf-objdump"

DEFAULT_OPTS = ["O0", "O1", "O2", "O3"]
DEFAULT_MAX_N = 6
DEFAULT_CYCLES = 20_000_000

TRACED_SCALAR_RE = re.compile(r"Traced (\d+) instructions")
TRACED_SS_RE = re.compile(r"Traced (\d+) superscalar cycles \((\d+) instructions, issue width (\d+)\)")


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def detect_tile_size(src_path):
    try:
        with open(src_path) as f:
            content = f.read()
        m = re.search(r"#define\s+MAX_DIM\s+(\d+)U", content)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 16


def build_tracer(workdir, tile_dim=16):
    tracer_path = os.path.join(workdir, f"tracer_t{tile_dim}")
    srcs = [TRACE_HARNESS_SRC]
    for name in ("CPU.cpp", "memory.cpp", "NPU.cpp", "NPU_print.cpp"):
        path = os.path.join(ISA_DIR, "src", name)
        if os.path.exists(path):
            srcs.append(path)
    run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", os.path.join(ISA_DIR, "include"),
         f"-DNPU_MAX_DIM={tile_dim}", "-O2", "-o", tracer_path] + srcs)
    return tracer_path


def compile_for_opt(src, opt, workdir):
    """Cross-compiles `src` bare-metal at -<opt>, returns (bin_path, dis_path)."""
    elf_path = os.path.join(workdir, f"prog_{opt}.elf")
    bin_path = os.path.join(workdir, f"prog_{opt}.bin")
    dis_path = os.path.join(workdir, f"prog_{opt}.dis.txt")
    run([RISCV_CC, "-march=rv32im", "-mabi=ilp32", "-nostdlib", "-nostartfiles",
         "-ffreestanding", f"-{opt}", "-T", LINK_LD, "-o", elf_path, START_S, src])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    dis = run([RISCV_OBJDUMP, "-d", elf_path]).stdout
    with open(dis_path, "w") as f:
        f.write(dis)
    return bin_path, dis_path


def trace(tracer_path, bin_path, dis_path, out_path, cycles, mode, width):
    """Runs the tracer, returns (halted_normally, stderr_summary_line)."""
    result = run([tracer_path, bin_path, dis_path, out_path, str(cycles), mode, str(width)])
    summary = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
    return ("HIT CYCLE CAP" not in summary), summary


def parse_scalar_trace(path):
    """Scalar trace: one instruction per line -- cycles == instructions."""
    cycles = 0
    with open(path) as f:
        next(f, None)  # header
        for _ in f:
            cycles += 1
    return cycles


def parse_superscalar_trace(path):
    """Superscalar trace: one CYCLE per line, `issued` is column index 2.
    Returns (cycles, instructions, histogram {issued_count: num_cycles})."""
    cycles = 0
    instructions = 0
    hist = {}
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            issued = int(parts[2])
            hist[issued] = hist.get(issued, 0) + 1
            cycles += 1
            instructions += issued
    return cycles, instructions, hist


def analyze(src, opts, max_n, cycles_cap, keep_traces, dest_name, tile_size=None):
    if tile_size is None:
        tile_size = detect_tile_size(src)
    all_halted = True
    report = {}  # opt -> {"total_instr": int, "runs": {label: {...}}}

    with tempfile.TemporaryDirectory() as workdir:
        print(f"Building tracer (tile={tile_size}x{tile_size})...", file=sys.stderr)
        tracer_path = build_tracer(workdir, tile_dim=tile_size)

        for opt in opts:
            print(f"Compiling at -{opt}...", file=sys.stderr)
            bin_path, dis_path = compile_for_opt(src, opt, workdir)

            runs = {}
            total_instr = None

            def do_run(label, mode, width):
                nonlocal total_instr, all_halted
                out_path = os.path.join(workdir, f"{dest_name}_{opt}_{label}.txt")
                halted, summary = trace(tracer_path, bin_path, dis_path, out_path, cycles_cap, mode, width)
                if not halted:
                    all_halted = False
                    print(f"  WARNING: -{opt} {label} hit the cycle cap -- did not halt: {summary}", file=sys.stderr)
                if mode == "scalar":
                    cycles = parse_scalar_trace(out_path)
                    instructions = cycles
                    hist = {1: cycles}
                else:
                    cycles, instructions, hist = parse_superscalar_trace(out_path)
                if total_instr is None:
                    total_instr = instructions
                elif instructions != total_instr:
                    print(f"  NOTE: -{opt} {label} instruction count ({instructions}) "
                          f"differs from this level's baseline ({total_instr})", file=sys.stderr)
                runs[label] = {"cycles": cycles, "instructions": instructions, "hist": hist}
                if keep_traces:
                    os.makedirs(BUILD_DIR, exist_ok=True)
                    dest = os.path.join(BUILD_DIR, f"{dest_name}_{opt}_{label}.txt")
                    with open(out_path, "rb") as sf, open(dest, "wb") as df:
                        df.write(sf.read())

            do_run("scalar", "scalar", 0)
            for n in range(1, max_n + 1):
                do_run(f"n{n}", "superscalar", n)

            report[opt] = {"total_instructions": total_instr, "runs": runs}

    return report, all_halted


def print_report(report, max_n):
    scalar_label = "scalar"
    print()
    print("=" * 78)
    print("SUPERSCALAR ISSUE-WIDTH ANALYSIS")
    print("=" * 78)

    for opt, data in report.items():
        total = data["total_instructions"]
        runs = data["runs"]
        scalar_cycles = runs[scalar_label]["cycles"]
        print(f"\n--- {opt}  (total instructions = {total:,}) ---")
        print(f"{'n':>4}  {'cycles':>12}  {'avg IPC':>9}  {'speedup':>9}")
        print(f"{'scalar':>4}  {scalar_cycles:>12,}  {'1.0000':>9}  {'1.000x':>9}")
        prev_cycles = None
        converged_at = None
        for n in range(1, max_n + 1):
            r = runs[f"n{n}"]
            cycles = r["cycles"]
            avg_ipc = r["instructions"] / cycles
            speedup = scalar_cycles / cycles
            marker = ""
            if prev_cycles is not None and cycles == prev_cycles and converged_at is None:
                converged_at = n - 1
            prev_cycles = cycles
            print(f"{n:>4}  {cycles:>12,}  {avg_ipc:>9.4f}  {speedup:>8.3f}x{marker}")
        if converged_at:
            print(f"  -> converges at n={converged_at} (identical cycle count from there on)")

        print(f"\n  Distribution of instructions co-issued per cycle:")
        for n in range(1, max_n + 1):
            r = runs[f"n{n}"]
            cycles = r["cycles"]
            parts = []
            for k in range(1, n + 1):
                cnt = r["hist"].get(k, 0)
                pct = 100 * cnt / cycles if cycles else 0
                parts.append(f"{k}:{cnt:,} ({pct:.1f}%)")
            print(f"    n={n}: " + "  ".join(parts))

    if len(report) > 1:
        print(f"\n--- Cross-level summary (best speedup, at converged n) ---")
        print(f"{'level':>6}  {'total instr':>12}  {'best speedup':>13}")
        for opt, data in report.items():
            runs = data["runs"]
            scalar_cycles = runs[scalar_label]["cycles"]
            best_speedup = max(scalar_cycles / runs[f"n{n}"]["cycles"] for n in range(1, max_n + 1))
            print(f"{opt:>6}  {data['total_instructions']:>12,}  {best_speedup:>12.3f}x")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(BUILD_DIR, "matmul_256_t2.c"),
                         help="bare-metal C source to analyze (default: build/matmul_256_t2.c)")
    parser.add_argument("--tile-size", type=int, default=None,
                         help="NPU native tile dimension (default: auto-detect from src or 16)")
    parser.add_argument("--opts", nargs="+", default=DEFAULT_OPTS,
                         help="optimization levels to sweep, e.g. O0 O1 O2 O3 (default: all four)")
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N,
                         help=f"sweep superscalar issue width from 1 to this (default: {DEFAULT_MAX_N})")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES,
                         help=f"cycle cap per run (default: {DEFAULT_CYCLES})")
    parser.add_argument("--no-keep-traces", action="store_true",
                         help="don't save the .txt traces to build/ -- only print the report")
    parser.add_argument("--json", default=None, help="also write the computed stats as JSON to this path")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isfile(src):
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1
    dest_name = os.path.splitext(os.path.basename(src))[0]

    report, all_halted = analyze(src, args.opts, args.max_n, args.cycles,
                                  keep_traces=not args.no_keep_traces, dest_name=dest_name,
                                  tile_size=args.tile_size)
    print_report(report, args.max_n)

    if args.json:
        # Histogram keys are ints -- JSON needs string keys.
        serializable = {
            opt: {
                "total_instructions": data["total_instructions"],
                "runs": {
                    label: {
                        "cycles": r["cycles"],
                        "instructions": r["instructions"],
                        "hist": {str(k): v for k, v in r["hist"].items()},
                    }
                    for label, r in data["runs"].items()
                },
            }
            for opt, data in report.items()
        }
        with open(args.json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Wrote {args.json}", file=sys.stderr)

    return 0 if all_halted else 1


if __name__ == "__main__":
    sys.exit(main())
