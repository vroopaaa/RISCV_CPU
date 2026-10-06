#!/usr/bin/env python3
"""
SIMT kernel correctness driver -- same methodology as
32I/tests/npu/run_npu_tests.py: for each kernel in c_tests/, cross-compile
it bare-metal for RV32IM (reusing tests/python/link.ld, but a SIMT-specific
start_simt.s -- see its header comment for why a per-lane stack is
mandatory), run it through tests/simt/harness.cpp (loads the binary, drives
SIMTCore's single-warp lane-broadcast engine instead of the scalar CPU,
dumps a memory region to a file), and diff that dump against an
independently-computed expected result -- not trusting the core's own
output, same as the NPU harness.

Run from anywhere -- paths are resolved relative to this script's location.

Usage:
  python3 run_simt_tests.py                  # run every kernel below
  python3 run_simt_tests.py --kernels leaky_relu_32 tmc_masking
  python3 run_simt_tests.py --cycles 2000     # override the cycle budget
"""
import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/simt -> tests -> 32I

C_TESTS_DIR = os.path.join(SCRIPT_DIR, "c_tests")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
START_S = os.path.join(SCRIPT_DIR, "start_simt.s")  # SIMT-specific -- per-lane stack, see its header
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")
HARNESS = os.path.join(PROJECT_ROOT, "build", "simt_harness")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_FLAGS = ["-march=rv32im", "-mabi=ilp32", "-nostdlib", "-nostartfiles", "-ffreestanding", "-O0"]

RESULT_BASE = 0x100000
DEFAULT_CYCLES = 2000


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result


def to_signed32(v):
    """Memory::dump_range (and every SIMTCore register) stores raw unsigned
    32-bit words (two's complement for negatives) -- interpret back as
    signed before comparing/printing, same convention
    simulation/run_leaky_relu.py's to_signed32 uses."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def div_trunc(a, b):
    """Truncating-toward-zero division, matching RV32IM's real DIV (and
    CPU::alu_exec's case 0x4) -- NOT Python's floor-dividing `//`."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def parse_dump_file(path):
    """Reads Memory::dump_range's '0xADDR: VALUE' lines, returns signed
    int32 values in address order (see memory.cpp's write_range)."""
    values = []
    with open(path) as f:
        for line in f:
            m = re.match(r"0x([0-9a-fA-F]+):\s*([0-9a-fA-F]+)", line.strip())
            if m:
                values.append(to_signed32(int(m.group(2), 16)))
    return values


def build_kernel(name):
    c_path = os.path.join(C_TESTS_DIR, name + ".c")
    elf_path = os.path.join(BUILD_DIR, name + ".elf")
    bin_path = os.path.join(BUILD_DIR, name + ".bin")
    os.makedirs(BUILD_DIR, exist_ok=True)
    run([RISCV_CC, *RISCV_FLAGS, "-T", LINK_LD, "-o", elf_path, START_S, c_path])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return bin_path


def run_kernel(bin_path, cycles, dump_size, dump_path):
    run([HARNESS, bin_path, str(cycles), f"{RESULT_BASE:x}", f"{dump_size:x}", dump_path])


# ---------------------------------------------------------------------
# One entry per kernel: how many result words to dump, and how to compute
# the expected values independently (never by re-deriving them from the
# kernel's own source logic -- same arm's-length philosophy as the NPU
# harness computing A*B itself rather than trusting the emulator).
# ---------------------------------------------------------------------

def expected_scalar_multiply():
    vec = [2, 3, 5, 7]
    return [v * 4 for v in vec]


def expected_divergent_branch():
    return [100, 100] + [200] * 30  # lanes 0,1: idx<2; lanes 2..31: idx>=2


def expected_tmc_masking():
    return [7, 7] + [to_signed32(0xDEADBEEF)] * 30  # only lanes 0,1 narrowed in


def expected_leaky_relu_32():
    x = list(range(-16, 16))
    return [xi if xi >= 0 else div_trunc(xi, 8) for xi in x]


KERNELS = {
    "scalar_multiply":  (4,  expected_scalar_multiply),
    "divergent_branch": (32, expected_divergent_branch),
    "tmc_masking":      (32, expected_tmc_masking),
    "leaky_relu_32":    (32, expected_leaky_relu_32),
}


def run_one(name, cycles):
    vec_len, expected_fn = KERNELS[name]
    expected = expected_fn()
    dump_size = vec_len * 4

    print(f"\n=== {name} ===")
    bin_path = build_kernel(name)
    dump_path = os.path.join(BUILD_DIR, name + ".dump.txt")
    run_kernel(bin_path, cycles, dump_size, dump_path)
    actual = parse_dump_file(dump_path)[:vec_len]

    if len(actual) != len(expected):
        print(f"FAIL: dumped {len(actual)} words, expected {len(expected)}")
        return False

    mismatches = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if mismatches:
        print(f"FAIL: {len(mismatches)}/{len(expected)} mismatches")
        for i, e, a in mismatches[:20]:
            print(f"  lane {i}: expected {e}, got {a}")
        return False

    print(f"PASS: all {len(expected)} lanes bit-exact")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run SIMT kernel tests against the emulator.")
    parser.add_argument("--kernels", nargs="+", choices=sorted(KERNELS), default=sorted(KERNELS),
                         help="Which kernels to run (default: all)")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES, help="Cycle budget per kernel")
    args = parser.parse_args()

    all_passed = True
    for name in args.kernels:
        try:
            passed = run_one(name, args.cycles)
        except Exception as e:
            print(f"\n[{name}] Exception: {e}\n")
            passed = False
        all_passed = all_passed and passed

    print()
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
