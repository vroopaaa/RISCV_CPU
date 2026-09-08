#!/usr/bin/env python3
"""
NPU matmul test driver.

For a given square matrix size (a multiple of 16, since the NPU's native
tile is 16x16), this script:
  1. Generates random A/B matrices in Python.
  2. Fills them into a c_tests/matmul_template*.c's {{BIG_DIM}}/{{MATRIX_A_DATA}}/
     {{MATRIX_B_DATA}} placeholders, producing a real .c file. Two templates
     are available (--template): 1 is the original per-word MMIO tile
     load/store, 2 (default) uses the custom-0 strided tile-transfer
     instruction (opcode 0x0B) added to CPU.cpp instead.
  3. Cross-compiles it bare-metal for RV32IM (reusing tests/python/start.s
     and link.ld -- this test needs nothing test-specific from them, just a
     stack and a jump into main()).
  4. Runs the resulting binary on the emulator's shared harness
     (build/run_test, the same one tests/basic/main.cpp builds), asking it
     to dump the result region to a file via Memory::dump_range's file
     overload instead of stdout.
  5. Computes A*B independently in Python and diffs it word-for-word
     against what the emulator wrote out.

Run from anywhere -- paths are resolved relative to this script's location.

Usage:
  python3 run_npu_tests.py                          # default sizes: 16 32 256, random data
  python3 run_npu_tests.py --sizes 16 32 48 64       # specific size(s), space-separated
  python3 run_npu_tests.py --sizes 128               # just one size
  python3 run_npu_tests.py --seed 42                 # reproducible run (same data every time)
  python3 run_npu_tests.py --low -100 --high 100     # override the random value range (default -10..10)
  python3 run_npu_tests.py --sizes 512 --cycles 5000000   # override the cycle budget if auto-estimate isn't enough
  python3 run_npu_tests.py --template 1              # use the original per-word MMIO template instead
  python3 run_npu_tests.py --mode superscalar               # run the SAME correctness check through the
                                                              # superscalar issue path (see ../../../docs/
                                                              # superscalar.md) instead of the scalar pipeline --
                                                              # exercises the NPU-specific hazard rules
                                                              # (load_a/load_b/store_c co-issue) end to end,
                                                              # not just the hazard-scan scheduling logic
  python3 run_npu_tests.py --mode superscalar --issue-width 4   # pin the issue width (default: CPU's own max)
  python3 run_npu_tests.py --opt O2                          # compile at -O2 instead of the default -O0
  python3 run_npu_tests.py --opt O2 --mode superscalar --issue-width 4

Sizes must be multiples of 16; anything else raises a ValueError for that
size and moves on to the next. Exit code is 0 if everything passed, 1 if
anything failed.
"""

import argparse
import os
import random
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/npu -> tests -> 32I

TEMPLATE_PATHS = {
    1: os.path.join(SCRIPT_DIR, "c_tests", "matmul_template.c"),
    # Same BIG_DIM/MATRIX_A_DATA/MATRIX_B_DATA placeholders, but tile
    # load/store use the custom-0 strided tile-transfer instruction
    # (opcode 0x0B) instead of per-word npu_write32/npu_move_word loops.
    2: os.path.join(SCRIPT_DIR, "c_tests", "matmul_template_2.c"),
}
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

# Reused as-is: this test needs nothing test-specific (no signature symbols
# etc.), just a stack + jump into main(), which is all start.s/link.ld do.
START_S = os.path.join(PROJECT_ROOT, "tests", "python", "start.s")
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")

HARNESS = os.path.join(PROJECT_ROOT, "build", "run_test")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
# Optimization level is now a per-call argument (see build_riscv_bin) --
# these are just the flags that never change.
RISCV_FLAGS = [
    "-march=rv32im", "-mabi=ilp32",
    "-nostdlib", "-nostartfiles", "-ffreestanding",
]
DEFAULT_OPT = "O0"

DEFAULT_TILE_SIZE = 16
MAX_DIM = DEFAULT_TILE_SIZE  # for backward compatibility
# Must match matmul_template.c's RESULT_BASE_ADDR.
RESULT_BASE_ADDR = 0xC8000

# Rough cycle estimate: MMIO write/drain count (and therefore cycle count)
# scales roughly with BIG_DIM^3 for this tiled algorithm. Calibrated against
# a measured 231482 cycles at BIG_DIM=32 (after moving A/B to static const,
# eliminating a big memcpy-driven fixed cost), with a 3x safety margin on
# top -- this is a generous cap, not a tuned value, since the emulator
# halts on its own once the test finishes.
_CALIBRATION_DIM = 32
_CALIBRATION_CYCLES = 231482


def estimate_cycles(big_dim, tile_dim=DEFAULT_TILE_SIZE):
    tiles = big_dim // tile_dim
    # ~50 cycles per tile operation with generous safety headroom
    return int((tiles ** 3) * 50) + 500000


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def get_harness_path(tile_dim=DEFAULT_TILE_SIZE):
    return os.path.join(PROJECT_ROOT, "build", f"run_test_t{tile_dim}")


def build_harness(tile_dim=DEFAULT_TILE_SIZE):
    harness_path = get_harness_path(tile_dim)
    run(["make", f"build/run_test_t{tile_dim}"], cwd=PROJECT_ROOT)
    if not os.path.exists(harness_path):
        raise RuntimeError(f"Harness not found at {harness_path} after build")
    return harness_path


def generate_matrix(size, low, high, rng):
    return [[rng.randint(low, high) for _ in range(size)] for _ in range(size)]


def format_matrix_c(matrix, size):
    lines = []
    for row in matrix:
        lines.append("        " + ", ".join(str(v) for v in row) + ",")
    return "\n".join(lines)


def format_matrix_tiled_c(matrix, size, tile=DEFAULT_TILE_SIZE):
    """Same data as format_matrix_c, but reordered tile-major: each tile x
    tile block contiguous (row-major within the tile), tiles visited in
    row-major grid order. Matches matmul_template_2.c's A/B layout, which
    lets its npu_load_a/npu_load_b read a whole tile in one contiguous
    access instead of `tile` separate strided row reads."""
    tiles_per_side = size // tile
    lines = []
    for ti in range(tiles_per_side):
        for tj in range(tiles_per_side):
            for r in range(tile):
                row_vals = matrix[ti * tile + r][tj * tile: tj * tile + tile]
                lines.append("        " + ", ".join(str(v) for v in row_vals) + ",")
    return "\n".join(lines)


def render_test_c(big_dim, tile_dim, A, B, out_path, template_path, template):
    with open(template_path) as f:
        contents = f.read()

    # Template 2's A/B are tile-major (see matmul_template_2.c's header
    # comment); template 1 still expects plain row-major.
    formatter = (lambda m, sz: format_matrix_tiled_c(m, sz, tile=tile_dim)) if template == 2 else format_matrix_c

    filled = (
        contents
        .replace("{{BIG_DIM}}", str(big_dim))
        .replace("{{TILE_DIM}}", str(tile_dim))
        .replace("{{MATRIX_A_DATA}}", formatter(A, big_dim))
        .replace("{{MATRIX_B_DATA}}", formatter(B, big_dim))
    )

    with open(out_path, "w") as f:
        f.write(filled)


def build_riscv_bin(c_path, name, opt=DEFAULT_OPT):
    elf_path = os.path.join(BUILD_DIR, name + ".elf")
    bin_path = os.path.join(BUILD_DIR, name + ".bin")
    run([RISCV_CC, *RISCV_FLAGS, f"-{opt}", "-T", LINK_LD, "-o", elf_path, START_S, c_path])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return bin_path


def run_simulator(harness_path, bin_path, cycles, dump_base, dump_size, dump_path, mode, issue_width):
    """Returns the harness's own reported cycle count (from its Final State
    dump), or None if it couldn't be found."""
    cmd = [harness_path, bin_path, str(cycles), f"{dump_base:x}", f"{dump_size:x}", dump_path]
    if mode == "superscalar":
        cmd += [mode, str(issue_width)]
    result = run(cmd)
    matches = re.findall(r"Cycles:\s*(\d+)", result.stdout)
    return int(matches[-1]) if matches else None


def parse_dump_file(path):
    """Reads Memory::dump_range's '0xADDR: VALUE' lines (value has no '0x'
    prefix -- see write_range() in memory.cpp), returns signed int32 values
    in address order."""
    values = []
    with open(path) as f:
        for line in f:
            m = re.match(r"0x([0-9a-fA-F]+):\s*([0-9a-fA-F]+)", line.strip())
            if not m:
                continue
            val = int(m.group(2), 16)
            if val >= 0x80000000:
                val -= 0x100000000
            values.append(val)
    return values


def compute_expected(A, B, size):
    expected = []
    for i in range(size):
        for j in range(size):
            s = 0
            for k in range(size):
                s += A[i][k] * B[k][j]
            expected.append(s)
    return expected


def run_one(big_dim, seed, low, high, cycles, template, mode, issue_width, opt=DEFAULT_OPT,
            tile_size=DEFAULT_TILE_SIZE, harness=None):
    if big_dim % tile_size != 0:
        raise ValueError(f"big_dim must be a multiple of {tile_size}, got {big_dim}")

    if harness is None:
        harness = build_harness(tile_size)

    os.makedirs(BUILD_DIR, exist_ok=True)
    rng = random.Random(seed)

    A = generate_matrix(big_dim, low, high, rng)
    B = generate_matrix(big_dim, low, high, rng)

    name = f"matmul_{big_dim}_t{template}_s{tile_size}_{opt}"
    c_path = os.path.join(BUILD_DIR, name + ".c")
    render_test_c(big_dim, tile_size, A, B, c_path, TEMPLATE_PATHS[template], template)

    mode_desc = mode if mode == "scalar" else f"{mode} (issue width {issue_width or 'default'})"
    print(f"\n=== {big_dim}x{big_dim} (tile={tile_size}x{tile_size}, seed={seed}, template={template}, -{opt}, mode={mode_desc}) ===")
    bin_path = build_riscv_bin(c_path, name, opt)

    dump_size = big_dim * big_dim * 4
    dump_path = os.path.join(BUILD_DIR, name + ".dump.txt")
    actual_cycles = cycles if cycles is not None else estimate_cycles(big_dim, tile_size)
    ran_cycles = run_simulator(harness, bin_path, actual_cycles, RESULT_BASE_ADDR, dump_size, dump_path, mode, issue_width)
    if ran_cycles is not None:
        print(f"  ran in {ran_cycles:,} cycles")

    actual = parse_dump_file(dump_path)
    expected = compute_expected(A, B, big_dim)

    n = min(len(actual), len(expected))
    short = len(actual) < len(expected)
    if short:
        print(f"WARNING: dump only produced {len(actual)}/{len(expected)} words "
              f"(did it halt early / run out of cycles? cycles={actual_cycles})")

    mismatches = [(i, expected[i], actual[i]) for i in range(n) if expected[i] != actual[i]]

    if short or mismatches:
        print(f"FAIL: {len(mismatches)}/{n} mismatches")
        for i, exp, act in mismatches[:20]:
            print(f"  idx {i}: expected {exp}, got {act}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        return False

    print(f"PASS: all {n} results bit-exact")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run NPU matmul tests against the emulator.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 256],
                         help="Square matrix sizes to test, each a multiple of tile size (default: 16 32 256)")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                         help=f"NPU native tile dimension (default: {DEFAULT_TILE_SIZE}, e.g. 8, 16, 32, 64)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: random each run)")
    parser.add_argument("--low", type=int, default=-10, help="Minimum random element value")
    parser.add_argument("--high", type=int, default=10, help="Maximum random element value")
    parser.add_argument("--cycles", type=int, default=None,
                         help="Override the cycle budget (default: auto-estimated per size)")
    parser.add_argument("--template", type=int, choices=sorted(TEMPLATE_PATHS), default=2,
                         help="Which matmul_template.c to use: 1 (per-word MMIO, original) or "
                              "2 (strided custom-0 tile transfer, default)")
    parser.add_argument("--mode", choices=["scalar", "superscalar"], default="scalar",
                         help="Run the emulator's scalar pipeline (default) or the superscalar "
                              "issue path (see ../../../docs/superscalar.md)")
    parser.add_argument("--issue-width", type=int, default=0,
                         help="Superscalar fetch/issue window width (0 = CPU's own default). "
                              "Ignored in scalar mode.")
    parser.add_argument("--opt", choices=["O0", "O1", "O2", "O3", "Os", "Og", "Ofast"], default=DEFAULT_OPT,
                         help=f"C compiler optimization level (default: {DEFAULT_OPT})")
    args = parser.parse_args()

    print(f"Building simulator test harness for tile size {args.tile_size}x{args.tile_size}...")
    harness = build_harness(args.tile_size)

    all_passed = True
    for big_dim in args.sizes:
        try:
            passed = run_one(big_dim, args.seed, args.low, args.high, args.cycles, args.template,
                              args.mode, args.issue_width, args.opt, args.tile_size, harness)
        except Exception as e:
            print(f"\n[{big_dim}x{big_dim}] Exception: {e}\n")
            passed = False
        all_passed = all_passed and passed

    print()
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
