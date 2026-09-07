#!/usr/bin/env python3
"""
NPU matmul test driver, with the K-tile loop unrolled.

Same correctness check as run_npu_tests.py (generate random A/B, render into
a C template, cross-compile, run on the emulator, diff against an
independent Python matmul) -- reuses that script's machinery directly. The
one difference: matmul_template_2.c's inner K-tile loop

    for (uint32_t Kt = 0; Kt < TILES; Kt++) {
        npu_load_a(...); npu_load_b(...); npu_write32(MAC_ADDR, 1);
    }

ends in a `bne` every iteration, and hazard_scan()'s control-hazard rule
closes the superscalar issue window at every single branch, regardless of
issue width -- unlike a RAW or memory hazard, no window width routes around
it. This script flattens that loop by a configurable factor in Python before
it's ever compiled, so the branch itself is gone for whichever K-tiles get
unrolled, and re-runs the same correctness check plus (optionally) the
superscalar analysis to measure the effect.

Run from anywhere -- paths are resolved relative to this script's location.

Usage:
  python3 run_npu_tests_2.py                             # default sizes 16 32 256, fully unrolled K-loop
  python3 run_npu_tests_2.py --sizes 256                  # just one size
  python3 run_npu_tests_2.py --unroll 4                   # unroll factor 4 instead of full
  python3 run_npu_tests_2.py --unroll 1                   # no unrolling (sanity check vs. run_npu_tests.py)
  python3 run_npu_tests_2.py --mode superscalar --issue-width 4
  python3 run_npu_tests_2.py --seed 42 --sizes 256 --unroll 0 --mode superscalar --issue-width 6
  python3 run_npu_tests_2.py --opt O2 --unroll 0 --mode superscalar --issue-width 4

--unroll 0 (default) means "fully unroll" (factor = TILES, i.e. every
K-iteration flattened, no Kt loop left at all). Any other positive integer
is a partial unroll factor (clamped to TILES); if TILES isn't evenly
divisible by it, the leftover K-tiles run in a normal trailing loop.

Exit code is 0 if everything passed, 1 if anything failed.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_npu_tests as base  # reuse: generate_matrix, format_matrix_tiled_c, build_riscv_bin,
                               # run_simulator, parse_dump_file, compute_expected, build_harness, run()

SCRIPT_DIR = base.SCRIPT_DIR
BUILD_DIR = base.BUILD_DIR
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "c_tests", "matmul_template_2_unrolled.c")
MAX_DIM = base.MAX_DIM
RESULT_BASE_ADDR = base.RESULT_BASE_ADDR


def generate_kt_loop_body(tiles, unroll):
    """C source for the K-tile loop body (I, J, A, B, TILES, TILE_WORDS all
    in scope from the surrounding template), unrolled by `unroll`. A
    trailing remainder loop covers any leftover K-tiles if `tiles` isn't
    evenly divisible by `unroll`.

    Each unrolled block uses INCREMENTAL pointers (pA += TILE_WORDS,
    pB += TILES * TILE_WORDS) rather than recomputing an absolute address
    from a literal Kt each time. The first version of this unroller did the
    latter, and it backfired: with Kt as a compile-time constant, -O2 had to
    rebuild each tile's full address from scratch (a `lui`+`add` pair, the
    `add` RAW-dependent on the `lui` right before it, forcing its own lone
    cycle) instead of the single cheap register-register add the original
    loop used to advance a running pointer by a constant stride. That new
    per-copy hazard ate almost all of the cycle savings unrolling was
    supposed to buy by removing the K-loop's branch. Incremental pointers
    restore the original loop's addressing shape while still being
    branch-free straight-line code."""
    unroll = max(1, min(unroll, tiles))
    full_span = (tiles // unroll) * unroll
    lines = []

    def emit_incremental_block(count, kt_start_expr):
        lines.append(f"            {{")
        lines.append(f"                const int32_t* pA = &A[(I * TILES + ({kt_start_expr})) * TILE_WORDS];")
        lines.append(f"                const int32_t* pB = &B[(({kt_start_expr}) * TILES + J) * TILE_WORDS];")
        for i in range(count):
            if i > 0:
                lines.append(f"                pA += TILE_WORDS;")
                lines.append(f"                pB += TILES * TILE_WORDS;")
            lines.append(f"                npu_load_a(pA);")
            lines.append(f"                npu_load_b(pB);")
            lines.append(f"                npu_write32(MAC_ADDR, 1);")
        lines.append(f"            }}")

    if unroll == 1:
        lines.append(f"            for (uint32_t Kt = 0; Kt < {full_span}; Kt++) {{")
        lines.append(f"                npu_load_a(&A[(I * TILES + Kt) * TILE_WORDS]);")
        lines.append(f"                npu_load_b(&B[(Kt * TILES + J) * TILE_WORDS]);")
        lines.append(f"                npu_write32(MAC_ADDR, 1);")
        lines.append(f"            }}")
    elif full_span == unroll:
        # Fully flat for this whole tile -- no loop at all, unroll == tiles.
        emit_incremental_block(full_span, "0")
    else:
        lines.append(f"            for (uint32_t Kt = 0; Kt < {full_span}; Kt += {unroll}) {{")
        emit_incremental_block(unroll, "Kt")
        lines.append(f"            }}")

    if full_span < tiles:
        lines.append(f"            for (uint32_t Kt = {full_span}; Kt < TILES; Kt++) {{")
        emit_body("Kt")
        lines.append(f"            }}")

    return "\n".join(lines)


def render_test_c_unrolled(big_dim, A, B, unroll, out_path):
    tiles = big_dim // MAX_DIM
    kt_body = generate_kt_loop_body(tiles, unroll if unroll > 0 else tiles)

    with open(TEMPLATE_PATH) as f:
        contents = f.read()

    filled = (
        contents
        .replace("{{BIG_DIM}}", str(big_dim))
        .replace("{{MATRIX_A_DATA}}", base.format_matrix_tiled_c(A, big_dim))
        .replace("{{MATRIX_B_DATA}}", base.format_matrix_tiled_c(B, big_dim))
        .replace("{{KT_LOOP_BODY}}", kt_body)
    )

    with open(out_path, "w") as f:
        f.write(filled)


def run_one(big_dim, seed, low, high, cycles, unroll, mode, issue_width, opt=base.DEFAULT_OPT):
    if big_dim % MAX_DIM != 0:
        raise ValueError(f"big_dim must be a multiple of {MAX_DIM}, got {big_dim}")

    os.makedirs(BUILD_DIR, exist_ok=True)
    rng = random.Random(seed)

    A = base.generate_matrix(big_dim, low, high, rng)
    B = base.generate_matrix(big_dim, low, high, rng)

    tiles = big_dim // MAX_DIM
    effective_unroll = unroll if unroll > 0 else tiles
    unroll_desc = "full" if effective_unroll >= tiles else str(effective_unroll)
    name = f"matmul_{big_dim}_t2u_unroll{unroll_desc}_{opt}"
    c_path = os.path.join(BUILD_DIR, name + ".c")
    render_test_c_unrolled(big_dim, A, B, unroll, c_path)

    mode_desc = mode if mode == "scalar" else f"{mode} (issue width {issue_width or 'default'})"
    print(f"\n=== {big_dim}x{big_dim} (seed={seed}, K-unroll={unroll_desc}/{tiles}, -{opt}, mode={mode_desc}) ===")
    bin_path = base.build_riscv_bin(c_path, name, opt)

    dump_size = big_dim * big_dim * 4
    dump_path = os.path.join(BUILD_DIR, name + ".dump.txt")
    actual_cycles = cycles if cycles is not None else base.estimate_cycles(big_dim)
    ran_cycles = base.run_simulator(bin_path, actual_cycles, RESULT_BASE_ADDR, dump_size,
                                     dump_path, mode, issue_width)
    if ran_cycles is not None:
        print(f"  ran in {ran_cycles:,} cycles")

    actual = base.parse_dump_file(dump_path)
    expected = base.compute_expected(A, B, big_dim)

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 256],
                         help="Square matrix sizes to test, each a multiple of 16 (default: 16 32 256)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: random each run)")
    parser.add_argument("--low", type=int, default=-10, help="Minimum random element value")
    parser.add_argument("--high", type=int, default=10, help="Maximum random element value")
    parser.add_argument("--cycles", type=int, default=None,
                         help="Override the cycle budget (default: auto-estimated per size)")
    parser.add_argument("--unroll", type=int, default=0,
                         help="K-tile loop unroll factor (0 = fully unroll, the default; "
                              "1 = no unrolling, same shape as matmul_template_2.c)")
    parser.add_argument("--mode", choices=["scalar", "superscalar"], default="scalar",
                         help="Run the emulator's scalar pipeline (default) or the superscalar issue path")
    parser.add_argument("--issue-width", type=int, default=0,
                         help="Superscalar fetch/issue window width (0 = CPU's own default). "
                              "Ignored in scalar mode.")
    parser.add_argument("--opt", choices=["O0", "O1", "O2", "O3", "Os", "Og", "Ofast"], default=base.DEFAULT_OPT,
                         help=f"C compiler optimization level (default: {base.DEFAULT_OPT})")
    args = parser.parse_args()

    print("Building simulator test harness...")
    base.build_harness()

    all_passed = True
    for big_dim in args.sizes:
        try:
            passed = run_one(big_dim, args.seed, args.low, args.high, args.cycles,
                              args.unroll, args.mode, args.issue_width, args.opt)
        except Exception as e:
            print(f"\n[{big_dim}x{big_dim}] Exception: {e}\n")
            passed = False
        all_passed = all_passed and passed

    print()
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
