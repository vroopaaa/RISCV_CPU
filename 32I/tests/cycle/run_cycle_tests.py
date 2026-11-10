#!/usr/bin/env python3
"""Cycle counter tests: rdcycle / rdcycleh on the host CPU
(CRUDA/docs/design/cycle_counter.md).

Each program in programs/ is assembled, linked with tests/python/link.ld and
run on build/run_test (tests/basic/main.cpp, what emul runs) on the scalar
host and on the superscalar host at issue widths 1/2/4/10. The expected
register values are worked out by hand from the counter's rule: a read
returns the number of cycles completed before the cycle it issues in (the
start of its issue group on the superscalar host). The checks tie those
values to the harness's own final cycle count, so an off-by-one or a wrong
half shows up as a mismatch.

Usage: python3 run_cycle_tests.py      (build/run_test must be built: make build/run_test)
Exit code 0 if everything passed, 1 otherwise.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/cycle -> 32I
PROGRAMS = os.path.join(SCRIPT_DIR, "programs")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
GRID_TESTS = os.path.join(PROJECT_ROOT, "tests", "simt", "grid_tests")   # simt_macros.inc, identity.s
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")
RUN_TEST = os.path.join(PROJECT_ROOT, "build", "run_test")

CC = "riscv64-unknown-elf-gcc"
OBJCOPY = "riscv64-unknown-elf-objcopy"
NM = "riscv64-unknown-elf-nm"
# zicsr only so system_other.s can spell csrrw/csrr; rdcycle/rdcycleh assemble without it.
FLAGS = ["-march=rv32im_zicsr", "-mabi=ilp32", "-nostdlib", "-nostartfiles", "-ffreestanding",
         f"-Wa,-I{GRID_TESTS}"]

MODES = [("scalar", 0), ("superscalar", 1), ("superscalar", 2), ("superscalar", 4), ("superscalar", 10)]
BUDGET = 100000
MASK32 = 0xFFFFFFFF


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def build(name, extra=()):
    os.makedirs(BUILD_DIR, exist_ok=True)
    elf = os.path.join(BUILD_DIR, f"{name}.elf")
    binf = os.path.join(BUILD_DIR, f"{name}.bin")
    sources = [os.path.join(PROGRAMS, f"{name}.s")] + list(extra)
    run([CC, *FLAGS, "-T", LINK_LD, "-o", elf, *sources])
    run([OBJCOPY, "-O", "binary", elf, binf])
    syms = {}
    for line in run([NM, elf]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            syms[parts[2]] = int(parts[0], 16)
    return binf, syms


def simulate(binf, mode, width, start_cycle=0):
    dump = os.path.join(BUILD_DIR, "dump.txt")
    cmd = [RUN_TEST, binf, str(BUDGET), "100000", "4", dump, mode, str(width), str(start_cycle)]
    r = run(cmd)
    regs = {}
    for m in re.finditer(r"x(\d+):\s*0x([0-9a-fA-F]+)", r.stdout):
        regs[int(m.group(1))] = int(m.group(2), 16)      # last print (final state) wins
    cycles = int(re.findall(r"Cycles:\s*(\d+)", r.stdout)[-1])
    return regs, cycles, r.stderr


A0, A1, A2, A3 = 10, 11, 12, 13


def label(mode, width):
    return "scalar" if mode == "scalar" else f"w{width}"


def tail(mode, width, n):
    """Cycles taken by the last n instructions of a program (counting the
    halting jump) when they start a fresh fetch window: one each on the
    scalar host or at width 1, otherwise ceil(n / width) groups."""
    if mode == "scalar" or width == 1:
        return n
    return (n + width - 1) // width


def show(v):
    return f"{v:#x} ({v})" if isinstance(v, int) else repr(v)


def check(results, name, mode, width, want, got):
    ok = want == got
    tag = f"{name} [{label(mode, width)}]"
    if ok:
        print(f"PASS  {tag}")
    else:
        print(f"FAIL  {tag}")
        for k in want:
            if want[k] != got.get(k):
                print(f"      {k}: expected {show(want[k])}, got {show(got.get(k))}")
    results.append(ok)


def test_first(results):
    binf, _ = build("first")
    for mode, width in MODES:
        regs, _, err = simulate(binf, mode, width)
        check(results, "first instruction reads 0", mode, width,
              dict(a0=0, a1=0, stderr=""), dict(a0=regs[A0], a1=regs[A1], stderr=err))


def test_same_cycle(results):
    binf, _ = build("same_cycle")
    for mode, width in MODES:
        regs, _, err = simulate(binf, mode, width)
        want = 1 if (mode == "scalar" or width == 1) else 0
        check(results, "two reads back to back", mode, width,
              dict(a0=want, stderr=""), dict(a0=regs[A0], stderr=err))


def test_total(results):
    binf, _ = build("total")
    for mode, width in MODES:
        regs, cycles, err = simulate(binf, mode, width)
        # last two instructions: rdcycle a0, j _end
        want = (cycles - tail(mode, width, 2)) & MASK32
        check(results, "read before halt = final count - tail", mode, width,
              dict(a0=want, stderr=""), dict(a0=regs[A0], stderr=err))


def test_launch(results):
    binf, syms = build("launch", [os.path.join(GRID_TESTS, "identity.s")])
    per_warp = (syms["kernel_end"] - syms["kernel"]) // 4
    device = 2 * per_warp            # one block, 2 warps, both on SM 0, run one after the other
    for mode, width in MODES:
        regs, _, err = simulate(binf, mode, width)
        check(results, "LAUNCH device cycles included", mode, width,
              dict(a0=device + 2, stderr=""), dict(a0=regs[A0], stderr=err))


def test_wrap(results):
    start = (1 << 32) - 3
    binf, _ = build("wrap")
    for mode, width in MODES:
        regs, cycles, err = simulate(binf, mode, width, start)
        # first reads: rdcycleh a2 at `start`, rdcycle a3 one cycle later on
        # the scalar host / width 1, in the same group otherwise.
        a3_cycle = start + (1 if (mode == "scalar" or width == 1) else 0)
        # last reads: rdcycleh a1, rdcycle a0, j _end, starting a fresh window.
        if mode == "scalar" or width == 1:
            a1_cycle, a0_cycle = cycles - 3, cycles - 2
        elif width == 2:
            a1_cycle = a0_cycle = cycles - 2
        else:
            a1_cycle = a0_cycle = cycles - 1
        want = dict(a2=start >> 32, a3=a3_cycle & MASK32, a1=a1_cycle >> 32, a0=a0_cycle & MASK32,
                    wrapped=1, stderr="")
        got = dict(a2=regs[A2], a3=regs[A3], a1=regs[A1], a0=regs[A0],
                   wrapped=int(cycles >> 32 == 1), stderr=err)
        check(results, "64-bit read across the 2^32 wrap", mode, width, want, got)


def test_system_other(results):
    binf, syms = build("system_other")
    bad = ["bad_csrrw", "bad_csrrs", "bad_instret", "bad_ecall", "bad_ebreak"]
    for mode, width in MODES:
        regs, _, err = simulate(binf, mode, width)
        lines = [l for l in err.splitlines() if l.strip()]
        want = dict(a0=0x1234, a1=0x5678, a2=0x9abc, errors=len(bad))
        got = dict(a0=regs[A0], a1=regs[A1], a2=regs[A2], errors=len(lines))
        for b in bad:
            pc = f"pc {syms[b]:#x}"
            want[b] = 1
            got[b] = int(any(pc in l and "unsupported SYSTEM instruction" in l for l in lines))
        check(results, "other SYSTEM instructions: error, ignored", mode, width, want, got)
        if want != got:
            print("      stderr:\n" + "\n".join("        " + l for l in lines))


def main():
    if not os.path.exists(RUN_TEST):
        print(f"{RUN_TEST} not found -- build it first (make build/run_test)")
        return 1
    results = []
    test_first(results)
    test_same_cycle(results)
    test_total(results)
    test_launch(results)
    test_wrap(results)
    test_system_other(results)
    passed = sum(results)
    print(f"{passed}/{len(results)} passed")
    if passed == len(results):
        print("ALL CYCLE TESTS PASSED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
