#!/usr/bin/env python3
"""
Runs official riscv-arch-test unprivileged M-extension test files (DIV,
DIVU, MUL, MULH, MULHSU, MULHU, REM, REMU -- see TESTS below) against this
emulator, without the full ACT4 framework (Sail reference model, UDB config,
mise/uv toolchain -- see the conversation this was built in for why that's
out of scope here).

Instead:
  1. Builds the .S file twice against env/riscv_arch_test.h, a hand-written
     stand-in for the framework header that implements only the macros this
     test uses (see that file's docstring for why the full one isn't
     needed/usable here):
       - once normally, for this emulator's own dedicated harness
         (harness.cpp), which halts on this codebase's `j self` idiom and
         dumps the signature memory region;
       - once with -DQEMU_REFERENCE, which makes the *same* test body instead
         write its signature region to stdout via a Linux `write` syscall and
         exit, so it can run standalone under `qemu-riscv32` as a real,
         independently-implemented RISC-V reference.
     (An earlier version of this script hand-computed expected DIV results in
     Python by replaying the register/data-table flow -- that duplicated the
     CPU's own bookkeeping in a second, easy-to-desync way and produced false
     mismatches. Using QEMU as the reference avoids that class of bug.)
  2. Diffs the two signature dumps word-by-word.

Run from anywhere -- paths are resolved relative to this script's location.
"""

import os
import re
import struct
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # riscv-arch-test -> tests -> 32I

ENV_DIR = os.path.join(SCRIPT_DIR, "env")
# Test sources live under assembly_files/, mirroring the upstream riscv-arch-test
# repo's own tests/ layout (e.g. assembly_files/rv32i/M/M-div-00.S matches that
# repo's tests/rv32i/M/M-div-00.S) so a path here is recognizable at a glance.
ASSEMBLY_DIR = os.path.join(SCRIPT_DIR, "assembly_files")
# Shared by both the emulator-harness build and the QEMU_REFERENCE build --
# see link_arch_test.ld's own comment for why they must use the same one.
LINK_LD = os.path.join(SCRIPT_DIR, "link_arch_test.ld")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_NM = "riscv64-unknown-elf-nm"
QEMU_RISCV32 = "qemu-riscv32"
RISCV_FLAGS = [
    "-march=rv32im", "-mabi=ilp32",
    "-nostdlib", "-nostartfiles", "-ffreestanding", "-O0",
]

HARNESS_SRCS = [
    os.path.join(SCRIPT_DIR, "harness.cpp"),
    os.path.join(SCRIPT_DIR, "..", "basic", "loader.cpp"),
    os.path.join(PROJECT_ROOT, "src", "CPU.cpp"),
    os.path.join(PROJECT_ROOT, "src", "memory.cpp"),
    os.path.join(PROJECT_ROOT, "src", "NPU.cpp"),
    os.path.join(PROJECT_ROOT, "src", "NPU_print.cpp"),
]
HARNESS_BIN = os.path.join(BUILD_DIR, "arch_test_harness")

# Every .S file under ASSEMBLY_DIR is run automatically (see discover_tests())
# rather than listed by hand here -- copy a suite folder from the upstream
# repo's tests/ directory into assembly_files/ (preserving its relative path)
# and it's picked up without editing this file. DEFAULT_MAX_CYCLES is just a
# generous safety cap (the harness halts on its own once a test finishes; see
# CPU::is_halted()), not a tuned per-test value. Override here only if some
# future test genuinely needs more.
DEFAULT_MAX_CYCLES = 200000
MAX_CYCLES_OVERRIDES = {
    # "rv32i/I/I-some-huge-test-00.S": 500000,
}


def discover_tests():
    """Finds every .S file under ASSEMBLY_DIR, returning paths relative to it
    (matching the upstream repo's tests/<path> layout), sorted for stable output."""
    tests = []
    for root, _, files in os.walk(ASSEMBLY_DIR):
        for f in files:
            if f.endswith(".S"):
                tests.append(os.path.relpath(os.path.join(root, f), ASSEMBLY_DIR))
    return sorted(tests)


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result


def build_cpp_harness():
    run(["g++", "-Wall", "-Wextra", "-std=c++11", "-O2", "-o", HARNESS_BIN, *HARNESS_SRCS])


def build_riscv_elf(s_path, name):
    elf_path = os.path.join(BUILD_DIR, name + ".elf")
    run([RISCV_CC, *RISCV_FLAGS, "-I", ENV_DIR, "-T", LINK_LD, "-o", elf_path, s_path])
    return elf_path


def signature_bounds(elf_path):
    result = run([RISCV_NM, elf_path])
    start = end = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        addr, _, sym = parts[0], parts[1], parts[2]
        if sym == "begin_signature":
            start = int(addr, 16)
        elif sym == "end_signature":
            end = int(addr, 16)
    if start is None or end is None:
        raise RuntimeError(f"Could not find begin_signature/end_signature in {elf_path}")
    return start, end


def to_bin(elf_path, name):
    bin_path = os.path.join(BUILD_DIR, name + ".bin")
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return bin_path


def run_harness(bin_path, sig_start, sig_end, max_cycles):
    result = run([HARNESS_BIN, bin_path, f"{sig_start:x}", f"{sig_end:x}", str(max_cycles)])
    # harness.cpp's stdout is just hex words, but loader.cpp (shared with the
    # other test harnesses) also prints a "[Loader] Loaded ..." line to stdout.
    return [int(line, 16) for line in result.stdout.split() if re.fullmatch(r"[0-9a-fA-F]{8}", line)]


def build_qemu_reference_elf(s_path, name):
    """Cross-compile the same test body with QEMU_REFERENCE defined, so
    RVTEST_CODE_END writes the signature to stdout + exits instead of
    spinning forever. Uses the same LINK_LD as build_riscv_elf()."""
    elf_path = os.path.join(BUILD_DIR, name + "_qemu_ref.elf")
    run([RISCV_CC, *RISCV_FLAGS, "-DQEMU_REFERENCE", "-I", ENV_DIR,
         "-T", LINK_LD, "-o", elf_path, s_path])
    return elf_path


def run_qemu_reference(elf_path):
    """Runs the QEMU_REFERENCE build under qemu-riscv32 and returns the
    signature words it wrote to stdout via the write() syscall."""
    result = subprocess.run([QEMU_RISCV32, elf_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"qemu-riscv32 exited {result.returncode}\n--- stderr ---\n{result.stderr.decode(errors='replace')}"
        )
    raw = result.stdout
    if len(raw) % 4 != 0:
        raise RuntimeError(f"qemu-riscv32 wrote {len(raw)} bytes, not a multiple of 4")
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    print("Building C++ signature-dump harness...")
    build_cpp_harness()

    all_passed = True

    for rel_path in discover_tests():
        max_cycles = MAX_CYCLES_OVERRIDES.get(rel_path, DEFAULT_MAX_CYCLES)
        # Build artifacts are named after the file only (not the full relative
        # path) since they all land flat in BUILD_DIR regardless of which
        # upstream suite subdirectory the source came from.
        name = os.path.splitext(os.path.basename(rel_path))[0]
        s_path = os.path.join(ASSEMBLY_DIR, rel_path)

        print(f"\n=== {rel_path} ===")
        elf_path = build_riscv_elf(s_path, name)
        sig_start, sig_end = signature_bounds(elf_path)
        bin_path = to_bin(elf_path, name)

        actual = run_harness(bin_path, sig_start, sig_end, max_cycles)

        qemu_elf_path = build_qemu_reference_elf(s_path, name)
        expected = run_qemu_reference(qemu_elf_path)

        n = min(len(actual), len(expected))
        print("Actual signature:", actual)
        print("Expected signature:", expected)

        mismatches = [(i, expected[i], actual[i]) for i in range(n) if expected[i] != actual[i]]

        print(f"Signature region: 0x{sig_start:08x}..0x{sig_end:08x} ({len(actual)} words dumped)")
        print(f"QEMU reference produced {len(expected)} words, compared {n}")

        if len(actual) < len(expected):
            print(f"WARNING: harness only produced {len(actual)} words (did it halt early / run out of cycles?)")
            all_passed = False

        if mismatches:
            all_passed = False
            print(f"FAIL: {len(mismatches)}/{n} mismatches")
            for i, exp, act in mismatches[:20]:
                print(f"  test {i}: expected 0x{exp:08x}, got 0x{act:08x}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches) - 20} more")
        else:
            print(f"PASS: all {n} results bit-exact")

    print()
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
