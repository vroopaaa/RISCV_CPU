#!/usr/bin/env python3
"""
Optimization-level sweep for the RISC-V CPU simulator.

Unlike run_tests.py (which always cross-compiles at -O0), this cross-compiles
each C test at -O0/-O1/-O2/-O3 and runs every build against the simulator,
comparing each against a single native (host, optimization-independent)
ground-truth result. Useful for exercising codegen that differs across
optimization levels -- e.g. a 64-bit multiply that GCC lowers to a generic
mul/mulhu-only sequence at -O0/-O1 but to direct mulh/mulhsu instructions at
-O2 and above.

Run from anywhere -- paths are resolved relative to this script's location.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

C_DIR = os.path.join(SCRIPT_DIR, "c_tests")
START_S = os.path.join(SCRIPT_DIR, "start.s")
LINK_LD = os.path.join(SCRIPT_DIR, "link.ld")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build", "opt")

HARNESS = os.path.join(PROJECT_ROOT, "build", "run_test")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_BASE_FLAGS = [
    "-march=rv32im", "-mabi=ilp32",
    "-nostdlib", "-nostartfiles", "-ffreestanding",
]

OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3"]

# Test list: one C filename (from c_tests/) per line in run_opt_tests_list.txt.
# Blank lines and '#' comments are ignored. The simulator halts on its own once
# a test program finishes (CPU::is_halted()), so MAX_CYCLES only needs to be a
# generous upper bound -- shared across every -O level, since optimized builds
# only ever need fewer actual cycles, never more.
TEST_LIST = os.path.join(SCRIPT_DIR, "run_opt_tests_list.txt")
MAX_CYCLES = 5000


def load_test_list(path):
    """Read one test filename per line; blank lines and '#' comments are ignored."""
    names = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    return names


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
    """Compile with libc + NATIVE_TEST, run it, return the printed integer.
    Optimization-independent: this is the ground truth every -O level is checked against."""
    exe_path = os.path.join(BUILD_DIR, name + ".native")
    run(["gcc", "-DNATIVE_TEST", "-O0", c_path, "-o", exe_path])
    result = run([exe_path])
    return int(result.stdout.strip())


def build_riscv_bin(c_path, name, opt):
    """Cross-compile bare-metal for RV32IM at the given -O level, return path to raw .bin."""
    tag = f"{name}_{opt.lstrip('-')}"
    elf_path = os.path.join(BUILD_DIR, tag + ".elf")
    bin_path = os.path.join(BUILD_DIR, tag + ".bin")
    run([RISCV_CC, *RISCV_BASE_FLAGS, opt, "-T", LINK_LD, "-o", elf_path, START_S, c_path])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return bin_path


def run_simulator(bin_path, cycles):
    """Run the harness against a .bin file, extract x10 and the cycle count from the FINAL state."""
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

    cycles_match = re.search(r"Cycles:\s*(\d+)", final_section)
    if not cycles_match:
        raise RuntimeError(f"Could not find Cycles in final state:\n{final_section}")
    cycles_executed = int(cycles_match.group(1))

    return value, cycles_executed


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    print("Building simulator test harness...")
    build_harness()

    test_names = load_test_list(TEST_LIST)

    rows = []
    all_passed = True

    for filename in test_names:
        name = os.path.splitext(filename)[0]
        c_path = os.path.join(C_DIR, filename)

        try:
            native_result = run_native(c_path, name)
        except Exception as e:
            print(f"\n[{filename}] native build failed: {e}\n")
            for opt in OPT_LEVELS:
                rows.append((filename, opt, "ERR", "ERR", "ERR", "ERROR"))
            all_passed = False
            continue

        for opt in OPT_LEVELS:
            try:
                bin_path = build_riscv_bin(c_path, name, opt)
                sim_result, sim_cycles = run_simulator(bin_path, MAX_CYCLES)
                passed = (native_result == sim_result)
                status = "PASS" if passed else "FAIL"
            except Exception as e:
                sim_result = "ERR"
                sim_cycles = "ERR"
                status = "ERROR"
                print(f"\n[{filename} @ {opt}] Exception: {e}\n")

            all_passed = all_passed and (status == "PASS")
            rows.append((filename, opt, native_result, sim_result, sim_cycles, status))

    # ---- Report ----
    print()
    name_width = max(len("Test"), max((len(f) for f, *_ in rows), default=0)) + 2
    print(f"{'Test':<{name_width}}{'Opt':<6}{'Native':<12}{'Simulator':<12}{'Cycles':<10}{'Result'}")
    print("-" * (name_width + 52))
    for filename, opt, native_result, sim_result, sim_cycles, status in rows:
        print(f"{filename:<{name_width}}{opt:<6}{str(native_result):<12}{str(sim_result):<12}{str(sim_cycles):<10}{status}")
    print("-" * (name_width + 52))
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
