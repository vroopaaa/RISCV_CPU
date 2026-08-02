#!/usr/bin/env python3
"""
Test suite for the RISC-V CPU simulator.

For each C test file, this script:
  1. Compiles it NATIVELY (with -DNATIVE_TEST) using the host's gcc, runs it,
     and captures the printed result -- this is the "ground truth" answer.
  2. Cross-compiles it BARE-METAL for RV32IM using start.s + link.ld,
     strips it to a raw binary with objcopy.
  3. Runs that binary on the C++ simulator harness and reads back the
     result from register x10 (a0) in the printed final CPU state.
  4. Compares native vs simulator results and reports PASS/FAIL.

Run from anywhere -- paths are resolved relative to this script's location.
"""

import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/python -> tests -> root

C_DIR = os.path.join(SCRIPT_DIR, "c_tests")
START_S = os.path.join(SCRIPT_DIR, "start.s")
LINK_LD = os.path.join(SCRIPT_DIR, "link.ld")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

HARNESS = os.path.join(PROJECT_ROOT, "build", "run_test")

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_FLAGS = [
    "-march=rv32im", "-mabi=ilp32",
    "-nostdlib", "-nostartfiles", "-ffreestanding", "-O0",
]

# ---------------------------------------------------------------------------
# Test registry: filename -> number of simulator cycles to allow
# (recursive/looping tests need more cycles than straight-line ones)
# ---------------------------------------------------------------------------
TESTS = {
    "mul_test.c": 200,
    "fib_test.c": 800,
    "fact_test.c": 1500,
}


def run(cmd, **kwargs):
    """Run a subprocess command, raising with useful output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def build_harness():
    """Make sure the simulator's test harness binary is built and current."""
    run(["make", "build/run_test"], cwd=PROJECT_ROOT)
    if not os.path.exists(HARNESS):
        raise RuntimeError(f"Harness not found at {HARNESS} after build")


def run_native(c_path, name):
    """Compile with libc + NATIVE_TEST, run it, return the printed integer."""
    exe_path = os.path.join(BUILD_DIR, name + ".native")
    run(["gcc", "-DNATIVE_TEST", "-O0", c_path, "-o", exe_path])
    result = run([exe_path])
    return int(result.stdout.strip())


def build_riscv_bin(c_path, name):
    """Cross-compile bare-metal for RV32IM, return path to raw .bin."""
    elf_path = os.path.join(BUILD_DIR, name + ".elf")
    bin_path = os.path.join(BUILD_DIR, name + ".bin")
    run([RISCV_CC, *RISCV_FLAGS, "-T", LINK_LD, "-o", elf_path, START_S, c_path])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return bin_path


def run_simulator(bin_path, cycles):
    """Run the harness against a .bin file, extract x10 from the FINAL state."""
    result = run([HARNESS, bin_path, str(cycles)])
    output = result.stdout

    if "Final State" not in output:
        raise RuntimeError(f"No 'Final State' section in harness output:\n{output}")
    final_section = output.split("Final State", 1)[1]

    match = re.search(r"x10:\s*0x([0-9a-fA-F]+)", final_section)
    if not match:
        raise RuntimeError(f"Could not find x10 in final state:\n{final_section}")

    value = int(match.group(1), 16)
    # Interpret as signed 32-bit, matching normal C `int` semantics
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    print("Building simulator test harness...")
    build_harness()

    rows = []
    all_passed = True

    for filename, cycles in TESTS.items():
        name = os.path.splitext(filename)[0]
        c_path = os.path.join(C_DIR, filename)

        try:
            native_result = run_native(c_path, name)
            bin_path = build_riscv_bin(c_path, name)
            sim_result = run_simulator(bin_path, cycles)
            passed = (native_result == sim_result)
            status = "PASS" if passed else "FAIL"
        except Exception as e:
            native_result = "ERR"
            sim_result = "ERR"
            status = "ERROR"
            print(f"\n[{filename}] Exception: {e}\n")

        all_passed = all_passed and (status == "PASS")
        rows.append((filename, native_result, sim_result, status))

    # ---- Report ----
    print()
    print(f"{'Test':<16}{'Native':<12}{'Simulator':<12}{'Result'}")
    print("-" * 52)
    for filename, native_result, sim_result, status in rows:
        print(f"{filename:<16}{str(native_result):<12}{str(sim_result):<12}{status}")
    print("-" * 52)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
