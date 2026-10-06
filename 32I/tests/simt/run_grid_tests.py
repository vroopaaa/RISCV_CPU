#!/usr/bin/env python3
"""
Grid-launch kernel tests -- same methodology as run_simt_tests.py /
tests/npu/run_npu_tests.py, but the kernels are hand-written ASSEMBLY
(grid_tests/*.s, SIMT ops from grid_tests/simt_macros.inc) so every
instruction the SMs run is visible in the source and the objdump listing.

For each (kernel, threads_per_block, num_blocks) case:
  1. assemble + link the kernel (plus a generated data file when it needs
     input data) with tests/python/link.ld, objcopy to a raw binary;
  2. look up the `kernel` entry (and data) symbols with nm;
  3. run build/grid_harness, which calls GridLauncher::launch_grid directly
     (no host CPU yet -- that is phase C) and dumps a memory region;
  4. diff the dump, the timed_out flag and the per-SM cycle counts against
     values computed independently here in Python.

Usage:
  python3 run_grid_tests.py                     # every case
  python3 run_grid_tests.py --kernels identity vec_add
"""
import argparse
import os
import random
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/simt -> 32I

KERNEL_DIR = os.path.join(SCRIPT_DIR, "grid_tests")
BUILD_DIR = os.path.join(KERNEL_DIR, "build")
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")
HARNESS = os.path.join(PROJECT_ROOT, "build", "grid_harness")

RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_OBJDUMP = "riscv64-unknown-elf-objdump"
RISCV_NM = "riscv64-unknown-elf-nm"
RISCV_FLAGS = ["-march=rv32im", "-mabi=ilp32", "-nostdlib", "-nostartfiles", "-ffreestanding",
               "-Wl,-e,kernel", f"-Wa,-I{KERNEL_DIR}"]

# Must match the hardware model (include/SIMT.h, include/GridLauncher.h, include/CPU.h).
NUM_SMS = 4
THREADS_PER_WARP = 32
WARPS_RESIDENT = 4
STACK_BYTES_PER_THREAD = 0x400
RAM_BYTES = 4 * 1024 * 1024
HOST_STACK_RESERVE = 0x10000
STACK_TOP = RAM_BYTES - HOST_STACK_RESERVE

RESULT_BASE = 0x100000
DEFAULT_MAX_ISSUES = 1000000
POISON = 0xDEADBEEF


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result


def parse_dump_file(path):
    """Memory::dump_range's '0xADDR: VALUE' lines -> unsigned 32-bit words in address order."""
    values = []
    with open(path) as f:
        for line in f:
            m = re.match(r"0x([0-9a-fA-F]+):\s*([0-9a-fA-F]+)", line.strip())
            if m:
                values.append(int(m.group(2), 16))
    return values


def build(name, extra_sources=()):
    os.makedirs(BUILD_DIR, exist_ok=True)
    elf = os.path.join(BUILD_DIR, name + ".elf")
    binf = os.path.join(BUILD_DIR, name + ".bin")
    run([RISCV_CC, *RISCV_FLAGS, "-T", LINK_LD, "-o", elf, os.path.join(KERNEL_DIR, name + ".s"), *extra_sources])
    run([RISCV_OBJCOPY, "-O", "binary", elf, binf])
    with open(os.path.join(BUILD_DIR, name + ".dis"), "w") as f:  # listing for RTL comparison
        f.write(run([RISCV_OBJDUMP, "-d", elf]).stdout)
    symbols = {}
    for line in run([RISCV_NM, elf]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            symbols[parts[2]] = int(parts[0], 16)
    return binf, symbols


def launch(binf, entry, tpb, nb, args, dump_base, dump_words, max_issues=DEFAULT_MAX_ISSUES, tag="run"):
    dump_path = os.path.join(BUILD_DIR, f"{tag}.dump.txt")
    out = run([HARNESS, binf, f"{entry:x}", str(tpb), str(nb), f"{args:x}", str(max_issues),
               f"{dump_base:x}", f"{dump_words * 4:x}", dump_path]).stdout
    m = re.search(r"cycles=(\d+) timed_out=(\d) sm_cycles=([\d,]+)", out)
    if not m:
        raise RuntimeError(f"harness output not understood:\n{out}")
    return {
        "cycles": int(m.group(1)),
        "timed_out": m.group(2) == "1",
        "sm_cycles": [int(x) for x in m.group(3).split(",")],
        "dump": parse_dump_file(dump_path)[:dump_words],
    }


# ---------------------------------------------------------------------
# Independent model of the hardware mapping (never read back from the emulator).
# ---------------------------------------------------------------------

def block_sm(b):
    return b % NUM_SMS  # round-robin


def hw_tid(b, t):
    return (block_sm(b) * WARPS_RESIDENT + t // THREADS_PER_WARP) * THREADS_PER_WARP + t % THREADS_PER_WARP


def lane_sp(b, t):
    return STACK_TOP - (hw_tid(b, t) + 1) * STACK_BYTES_PER_THREAD


def warps_per_block(tpb):
    return (tpb + THREADS_PER_WARP - 1) // THREADS_PER_WARP


def compare(what, expected, actual):
    bad = [(i, e, a) for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if len(actual) != len(expected):
        bad.append((len(actual), f"{len(expected)} words", f"{len(actual)} words"))
    for i, e, a in bad[:10]:
        print(f"    {what}[{i}]: expected {e:#x}, got {a:#x}" if isinstance(e, int) else f"    {what}: expected {e}, got {a}")
    return not bad


# ---------------------------------------------------------------------
# Kernels. Each returns True on pass.
# ---------------------------------------------------------------------

def test_identity(tpb, nb):
    binf, sym = build("identity")
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, 5 * total, tag=f"identity_{tpb}x{nb}")
    expected = [0] * (5 * total)
    for b in range(nb):
        for t in range(tpb):
            gid = b * tpb + t
            for f, v in enumerate([b, tpb, nb, t, hw_tid(b, t)]):
                expected[f * total + gid] = v
    ok = compare("out", expected, r["dump"])
    # Straight-line kernel: every warp issues exactly (kernel_end - kernel)/4 instructions.
    kernel_len = (sym["kernel_end"] - sym["kernel"]) // 4
    want_sm = [0] * NUM_SMS
    for b in range(nb):
        want_sm[block_sm(b)] += warps_per_block(tpb) * kernel_len
    ok &= compare("sm_cycles", want_sm, r["sm_cycles"])
    ok &= compare("cycles", [max(want_sm)], [r["cycles"]])
    ok &= compare("timed_out", [0], [int(r["timed_out"])])
    return ok


def test_stack(tpb, nb):
    binf, sym = build("stack")
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, 2 * total, tag=f"stack_{tpb}x{nb}")
    expected = [0] * (2 * total)
    for b in range(nb):
        for t in range(tpb):
            gid = b * tpb + t
            expected[gid] = (gid * gid + gid + 1) & 0xFFFFFFFF
            expected[total + gid] = lane_sp(b, t)
    return compare("out", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])])


def test_vec_add(tpb, nb, n, seed=1):
    total = tpb * nb
    rng = random.Random(seed)
    a = [rng.randrange(0, 1 << 32) for _ in range(total)]
    b = [rng.randrange(0, 1 << 32) for _ in range(total)]
    data = os.path.join(BUILD_DIR, f"vec_add_data_{tpb}x{nb}.s")
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(data, "w") as f:
        f.write("    .data\n    .globl vec_args\n    .globl vec_c\n    .align 2\n")
        f.write(f"vec_args: .word vec_a, vec_b, vec_c, {n}\n")
        f.write("vec_a: .word " + ", ".join(str(x) for x in a) + "\n")
        f.write("vec_b: .word " + ", ".join(str(x) for x in b) + "\n")
        f.write("vec_c: .word " + ", ".join([str(POISON)] * total) + "\n")
    binf, sym = build("vec_add", [data])
    r = launch(binf, sym["kernel"], tpb, nb, sym["vec_args"], sym["vec_c"], total, tag=f"vec_add_{tpb}x{nb}")
    expected = [((a[i] + b[i]) & 0xFFFFFFFF) if i < n else POISON for i in range(total)]
    return compare("c", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])])


def test_divergent(tpb, nb):
    binf, sym = build("divergent")
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, total, tag=f"divergent_{tpb}x{nb}")
    expected = []
    for gid in range(total):
        t = gid % tpb
        if t % 2 == 1:
            expected.append(gid * 3 + (1000 if t < 16 else 0))
        else:
            expected.append(gid + 100)
    return compare("out", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])])


def test_runaway(tpb, nb, cap=500):
    binf, sym = build("runaway")
    r = launch(binf, sym["kernel"], tpb, nb, 0, RESULT_BASE, 1, max_issues=cap, tag=f"runaway_{tpb}x{nb}")
    # Block 0 (on SM 0) burns the whole cap; the rest of the grid is abandoned.
    want_sm = [cap] + [0] * (NUM_SMS - 1)
    return compare("timed_out", [1], [int(r["timed_out"])]) & compare("sm_cycles", want_sm, r["sm_cycles"])


# (kernel, threads_per_block, num_blocks, extra args) -- block counts wrap NUM_SMS,
# thread counts cover a single lane, one full warp, a partial last warp and a full block.
CASES = [
    ("identity",  1,   3,  {}),
    ("identity",  32,  1,  {}),
    ("identity",  40,  4,  {}),
    ("identity",  40,  5,  {}),
    ("identity",  128, 9,  {}),
    ("stack",     40,  5,  {}),
    ("stack",     128, 4,  {}),
    ("vec_add",   40,  5,  {"n": 190}),   # last block partly out of range
    ("vec_add",   128, 9,  {"n": 1000}),
    ("divergent", 32,  1,  {}),
    ("divergent", 40,  5,  {}),
    ("runaway",   32,  8,  {}),
]
TESTS = {"identity": test_identity, "stack": test_stack, "vec_add": test_vec_add,
         "divergent": test_divergent, "runaway": test_runaway}


def main():
    parser = argparse.ArgumentParser(description="Run grid-launch assembly kernel tests.")
    parser.add_argument("--kernels", nargs="+", choices=sorted(TESTS), default=sorted(TESTS))
    args = parser.parse_args()

    passed = failed = 0
    for name, tpb, nb, extra in CASES:
        if name not in args.kernels:
            continue
        label = f"{name} {tpb} threads x {nb} blocks"
        try:
            ok = TESTS[name](tpb, nb, **extra)
        except Exception as e:
            print(f"  [{label}] Exception: {e}")
            ok = False
        print(f"{'PASS' if ok else 'FAIL'}: {label}")
        passed += ok
        failed += not ok

    print(f"\n{passed}/{passed + failed} passed")
    print("ALL GRID TESTS PASSED" if failed == 0 else "SOME GRID TESTS FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
