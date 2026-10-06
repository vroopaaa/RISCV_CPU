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


def build(name, extra_sources=(), sources=None, defsyms=None):
    """sources: .s files in grid_tests/ in link order (default: just name.s);
    the first one's .text lands at address 0, where the host CPU starts."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    elf = os.path.join(BUILD_DIR, name + ".elf")
    binf = os.path.join(BUILD_DIR, name + ".bin")
    srcs = [os.path.join(KERNEL_DIR, f + ".s") for f in (sources or [name])]
    syms = [f"-Wa,--defsym,{k}={v}" for k, v in (defsyms or {}).items()]
    run([RISCV_CC, *RISCV_FLAGS, *syms, "-T", LINK_LD, "-o", elf, *srcs, *extra_sources])
    run([RISCV_OBJCOPY, "-O", "binary", elf, binf])
    with open(os.path.join(BUILD_DIR, name + ".dis"), "w") as f:  # listing for RTL comparison
        f.write(run([RISCV_OBJDUMP, "-d", elf]).stdout)
    symbols = {}
    for line in run([RISCV_NM, elf]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            symbols[parts[2]] = int(parts[0], 16)
    return binf, symbols


def launch(binf, entry, tpb, nb, args, dump_base, dump_words, max_issues=DEFAULT_MAX_ISSUES, tag="run",
           ram=RAM_BYTES, reserve=HOST_STACK_RESERVE):
    dump_path = os.path.join(BUILD_DIR, f"{tag}.dump.txt")
    proc = run([HARNESS, binf, f"{entry:x}", str(tpb), str(nb), f"{args:x}", str(max_issues),
                f"{dump_base:x}", f"{dump_words * 4:x}", dump_path, f"{ram:x}", f"{reserve:x}"])
    out = proc.stdout
    stderr = proc.stderr
    m = re.search(r"cycles=(\d+) timed_out=(\d) faulted=(\d) sm_cycles=([\d,]+)", out)
    if not m:
        raise RuntimeError(f"harness output not understood:\n{out}")
    return {
        "cycles": int(m.group(1)),
        "timed_out": m.group(2) == "1",
        "faulted": m.group(3) == "1",
        "sm_cycles": [int(x) for x in m.group(4).split(",")],
        "dump": parse_dump_file(dump_path)[:dump_words],
        "stderr": stderr,
    }


def run_host(binf, dump_base, dump_words, tag, max_cycles=5000000, attach_gpu=True):
    """Boots the host CPU at pc 0 with the GPU attached; the host program
    launches kernels itself through the LAUNCH opcode (0x5B)."""
    dump_path = os.path.join(BUILD_DIR, f"{tag}.dump.txt")
    cmd = [HARNESS, "--host", binf, str(max_cycles), f"{dump_base:x}", f"{dump_words * 4:x}", dump_path]
    if not attach_gpu:
        cmd.append("no-gpu")
    proc = run(cmd)
    out = proc.stdout
    m = re.search(r"halted=(\d) cpu_cycles=(\d+) sm_cycles=([\d,]+)", out)
    if not m:
        raise RuntimeError(f"harness output not understood:\n{out}")
    return {
        "halted": m.group(1) == "1",
        "cpu_cycles": int(m.group(2)),
        "sm_cycles": [int(x) for x in m.group(3).split(",")],
        "dump": parse_dump_file(dump_path)[:dump_words],
        "stderr": proc.stderr,
    }


def read_word(binf_dump, addr, base):
    return binf_dump[(addr - base) // 4]


# ---------------------------------------------------------------------
# Independent model of the hardware mapping (never read back from the emulator).
# ---------------------------------------------------------------------

def block_sm(b):
    return b % NUM_SMS  # round-robin


def hw_tid(b, t):
    return (block_sm(b) * WARPS_RESIDENT + t // THREADS_PER_WARP) * THREADS_PER_WARP + t % THREADS_PER_WARP


def lane_sp(b, t, stack_top=STACK_TOP):
    return stack_top - (hw_tid(b, t) + 1) * STACK_BYTES_PER_THREAD


def warps_per_block(tpb):
    return (tpb + THREADS_PER_WARP - 1) // THREADS_PER_WARP


def check_stderr(r, expected=()):
    """No error output unless the test expects it: each expected message must
    appear, and no other line may (a stray [Memory Error] from an
    out-of-range access fails the test even if the dump happens to match)."""
    lines = [l for l in r["stderr"].splitlines() if l.strip()]
    ok = True
    for msg in expected:
        if not any(msg in l for l in lines):
            print(f"    stderr: expected a line containing {msg!r}, got {lines[:5]}")
            ok = False
    stray = [l for l in lines if not any(msg in l for msg in expected)]
    for l in stray[:5]:
        print(f"    stderr: unexpected {l!r}")
    return ok and not stray


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
    return ok & check_stderr(r)


def test_stack(tpb, nb, ram=RAM_BYTES, reserve=HOST_STACK_RESERVE):
    """ram/reserve: the host stack reserve is whatever the CPU side passes in,
    and the device stacks start right below it, wherever RAM ends."""
    binf, sym = build("stack")
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, 2 * total,
               tag=f"stack_{tpb}x{nb}_{ram:x}_{reserve:x}", ram=ram, reserve=reserve)
    expected = [0] * (2 * total)
    for b in range(nb):
        for t in range(tpb):
            gid = b * tpb + t
            expected[gid] = (gid * gid + gid + 1) & 0xFFFFFFFF
            expected[total + gid] = lane_sp(b, t, ram - reserve)
    return compare("out", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])]) & check_stderr(r)


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
    return compare("c", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])]) & check_stderr(r)


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
    return compare("out", expected, r["dump"]) & compare("timed_out", [0], [int(r["timed_out"])]) & check_stderr(r)


def test_runaway(tpb, nb, cap=500):
    binf, sym = build("runaway")
    r = launch(binf, sym["kernel"], tpb, nb, 0, RESULT_BASE, 1, max_issues=cap, tag=f"runaway_{tpb}x{nb}")
    # Block 0 (on SM 0) burns the whole cap; the rest of the grid is abandoned.
    want_sm = [cap] + [0] * (NUM_SMS - 1)
    return (compare("timed_out", [1], [int(r["timed_out"])]) & compare("sm_cycles", want_sm, r["sm_cycles"])
            & check_stderr(r, ["exceeded", "timed out -- abandoning"]))


MARKER_ADDR = 0x180000
MARKER = 0x600D


def expected_identity(tpb, nb):
    total = tpb * nb
    expected = [0] * (5 * total)
    for b in range(nb):
        for t in range(tpb):
            gid = b * tpb + t
            for f, v in enumerate([b, tpb, nb, t, hw_tid(b, t)]):
                expected[f * total + gid] = v
    return expected


def test_host_launch(tpb, nb, attach_gpu=True):
    """Host CPU launches identity over the grid, then writes a marker.
    CPU cycles = its own instructions (straight line _start.._end, plus the
    halting jump) + the launch's device cycles."""
    binf, sym = build("host_launch", sources=["host_launch", "identity"], defsyms={"TPB": tpb, "NB": nb})
    total = tpb * nb
    words = (MARKER_ADDR - RESULT_BASE) // 4 + 1   # output region .. marker
    r = run_host(binf, RESULT_BASE, words, tag=f"host_launch_{tpb}x{nb}_{int(attach_gpu)}", attach_gpu=attach_gpu)
    host_instrs = (sym["_end"] - sym["_start"]) // 4 + 1
    kernel_len = (sym["kernel_end"] - sym["kernel"]) // 4
    want_sm = [0] * NUM_SMS
    if attach_gpu:
        for b in range(nb):
            want_sm[block_sm(b)] += warps_per_block(tpb) * kernel_len
    out = expected_identity(tpb, nb) if attach_gpu else [0] * (5 * total)
    ok = compare("halted", [1], [int(r["halted"])])
    ok &= compare("out", out, r["dump"][:5 * total])
    ok &= compare("marker", [MARKER], [read_word(r["dump"], MARKER_ADDR, RESULT_BASE)])
    ok &= compare("sm_cycles", want_sm, r["sm_cycles"])
    ok &= compare("cpu_cycles", [host_instrs + max(want_sm)], [r["cpu_cycles"]])
    ok &= check_stderr(r, [] if attach_gpu else ["no GPU attached"])
    return ok


def test_host_no_gpu(tpb, nb):
    """LAUNCH with no GPU attached: error + no-op, the host carries on."""
    return test_host_launch(tpb, nb, attach_gpu=False)


def test_host_two_launches(tpb, nb, n, seed=2):
    total = tpb * nb
    rng = random.Random(seed)
    a = [rng.randrange(0, 1 << 32) for _ in range(total)]
    b = [rng.randrange(0, 1 << 32) for _ in range(total)]
    data = os.path.join(BUILD_DIR, f"two_launch_data_{tpb}x{nb}.s")
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(data, "w") as f:
        f.write("    .data\n    .globl vec_args\n    .globl vec_args2\n    .globl vec_c\n    .globl vec_d\n"
                "    .globl host_sum\n    .align 2\n")
        f.write(f"vec_args:  .word vec_a, vec_b, vec_c, {n}\n")
        f.write(f"vec_args2: .word vec_c, vec_b, vec_d, {n}\n")
        f.write("host_sum: .word 0\n")
        f.write("vec_a: .word " + ", ".join(str(x) for x in a) + "\n")
        f.write("vec_b: .word " + ", ".join(str(x) for x in b) + "\n")
        f.write("vec_c: .word " + ", ".join([str(POISON)] * total) + "\n")
        f.write("vec_d: .word " + ", ".join([str(POISON)] * total) + "\n")
    binf, sym = build("host_two_launches", [data], sources=["host_two_launches", "vec_add"],
                      defsyms={"TPB": tpb, "NB": nb})
    base = sym["host_sum"]
    words = (sym["vec_d"] - base) // 4 + total
    r = run_host(binf, base, words, tag=f"host_two_launches_{tpb}x{nb}")
    c = [((a[i] + b[i]) & 0xFFFFFFFF) if i < n else POISON for i in range(total)]
    d = [((a[i] + 2 * b[i]) & 0xFFFFFFFF) if i < n else POISON for i in range(total)]
    ok = compare("halted", [1], [int(r["halted"])])
    ok &= compare("c", c, r["dump"][(sym["vec_c"] - base) // 4:][:total])
    ok &= compare("d", d, r["dump"][(sym["vec_d"] - base) // 4:][:total])
    ok &= compare("host_sum", [sum(d[:n]) & 0xFFFFFFFF], [r["dump"][0]])
    ok &= check_stderr(r)
    return ok


def test_host_bad_launch(tpb, nb):
    """129 threads/block, 0 blocks and reserved funct3 1: all skipped, host continues."""
    binf, sym = build("host_bad_launch", sources=["host_bad_launch", "identity"])
    words = (MARKER_ADDR - RESULT_BASE) // 4 + 1
    r = run_host(binf, RESULT_BASE, words, tag="host_bad_launch")
    ok = compare("halted", [1], [int(r["halted"])])
    ok &= compare("out", [0] * 64, r["dump"][:64])
    ok &= compare("marker", [MARKER], [read_word(r["dump"], MARKER_ADDR, RESULT_BASE)])
    ok &= compare("sm_cycles", [0] * NUM_SMS, r["sm_cycles"])
    ok &= compare("cpu_cycles", [(sym["_end"] - sym["_start"]) // 4 + 1], [r["cpu_cycles"]])
    ok &= check_stderr(r, ["threads_per_block 129", "nBlocks 0", "reserved LAUNCH funct3 1"])
    return ok


# Instructions an SM lane can't run: (raw word, name the error message must give).
BAD_INSTRUCTIONS = {
    "launch":      (0x0000005B, "LAUNCH"),       # nested launch from inside a kernel
    "npu":         (0x0000000B, "NPU"),          # custom-0, GPU->NPU handoff not built
    "ecall":       (0x00000073, "SYSTEM"),       # no traps / CSRs anywhere
    "wspawn":      (0x0000102B, "SIMT_WSPAWN"),  # needs the phase 4 scheduler
    "bar":         (0x0000302B, "SIMT_BAR"),     # needs the phase 4 scheduler
    "ident_sel4":  (0x0800702B, "SIMT_IDENT"),   # IDENT selector 4 doesn't exist
    "load_f3":     (0x00003003, "LOAD"),         # funct3 3 (LD) isn't RV32
    "store_f3":    (0x00003023, "STORE"),        # funct3 3 (SD) isn't RV32
    "branch_f3":   (0x00002063, "BRANCH"),       # funct3 2 isn't a branch
    "zero_word":   (0x00000000, "unknown"),      # empty memory / jumped into data
}
FENCE = 0x0FF0000F


def test_bad_instr(tpb, nb, bad):
    """The first warp to reach the bad instruction faults: that block stops and
    the rest of the grid is abandoned. Only block 0's warp 0 ran (up to and
    including the bad instruction), so only its lanes wrote 1, and nobody wrote 2."""
    word, name = BAD_INSTRUCTIONS[bad]
    binf, sym = build("bad_instr", defsyms={"BAD_WORD": word})
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, total, tag=f"bad_instr_{bad}")
    first_warp = min(tpb, THREADS_PER_WARP)
    expected = [1] * first_warp + [0] * (total - first_warp)
    issued = (sym["bad_at"] - sym["kernel"]) // 4 + 1      # the faulting instruction counts as issued
    ok = compare("out", expected, r["dump"])
    ok &= compare("faulted", [1], [int(r["faulted"])])
    ok &= compare("timed_out", [0], [int(r["timed_out"])])
    ok &= compare("sm_cycles", [issued] + [0] * (NUM_SMS - 1), r["sm_cycles"])
    ok &= check_stderr(r, [f"SM 0 block 0 warp 0 pc {sym['bad_at']:#x}: unsupported instruction {word:#010x} ({name})",
                           "faulted -- abandoning"])
    return ok


def test_fence_ok(tpb, nb):
    """FENCE is a legal no-op on the GPU (memory never reorders in this model)."""
    binf, sym = build("bad_instr", defsyms={"BAD_WORD": FENCE})
    total = tpb * nb
    r = launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, total, tag="fence_ok")
    return (compare("out", [2] * total, r["dump"]) & compare("faulted", [0], [int(r["faulted"])])
            & check_stderr(r))


def test_host_fault(tpb, nb, kernel):
    """A launch that fails on the GPU (fault or timeout) halts the CPU right
    after the LAUNCH: the marker after it is never written."""
    defsyms = {"TPB": tpb, "NB": nb}
    if kernel == "bad_instr":
        defsyms["BAD_WORD"] = BAD_INSTRUCTIONS["launch"][0]
    binf, sym = build("host_fault", sources=["host_fault", kernel], defsyms=defsyms)
    words = (MARKER_ADDR - RESULT_BASE) // 4 + 1
    r = run_host(binf, RESULT_BASE, words, tag=f"host_fault_{kernel}")
    host_instrs = (sym["launch_at"] - sym["_start"]) // 4 + 1   # up to and including the LAUNCH
    gpu_msg = "unsupported instruction" if kernel == "bad_instr" else "exceeded"
    ok = compare("halted", [1], [int(r["halted"])])
    ok &= compare("marker", [0], [read_word(r["dump"], MARKER_ADDR, RESULT_BASE)])
    ok &= compare("cpu_cycles", [host_instrs + max(r["sm_cycles"])], [r["cpu_cycles"]])
    ok &= check_stderr(r, [gpu_msg, "abandoning", f"[CPU Error] LAUNCH at pc {sym['launch_at']:#x} failed on the GPU -- halting"])
    return ok


def refused(r):
    """A refused launch runs nothing: 0 cycles, every SM idle, output untouched."""
    return (compare("cycles", [0], [r["cycles"]]) & compare("sm_cycles", [0] * NUM_SMS, r["sm_cycles"])
            & compare("out", [0], r["dump"]) & check_stderr(r, ["[GridLauncher Error]"]))


def test_bad_launch(tpb, nb):
    binf, sym = build("identity")
    return refused(launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, 1, tag=f"bad_{tpb}x{nb}"))


def test_ram_too_small(tpb, nb, ram, reserve):
    """RAM that can't hold the host reserve plus every SM's device stacks is
    refused up front instead of handing out sp values outside RAM."""
    binf, sym = build("identity")
    return refused(launch(binf, sym["kernel"], tpb, nb, RESULT_BASE, RESULT_BASE, 1,
                          tag=f"small_{ram:x}_{reserve:x}", ram=ram, reserve=reserve))


DEVICE_STACKS = NUM_SMS * WARPS_RESIDENT * THREADS_PER_WARP * STACK_BYTES_PER_THREAD  # 512KB

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
    ("stack",     40,  5,  {"ram": 2 * 1024 * 1024, "reserve": 0x8000}),   # other RAM size + reserve
    ("stack",     40,  5,  {"ram": 0x180000, "reserve": 0x180000 - DEVICE_STACKS}),  # exactly fits
    ("vec_add",   40,  5,  {"n": 190}),   # last block partly out of range
    ("vec_add",   128, 9,  {"n": 1000}),
    ("divergent", 32,  1,  {}),
    ("divergent", 40,  5,  {}),
    ("runaway",   32,  8,  {}),
    # Refused launches (plan 4.1 validation + the RAM-size check).
    ("bad_launch", 0,   1,     {}),
    ("bad_launch", 129, 1,     {}),
    ("bad_launch", 32,  0,     {}),
    ("bad_launch", 32,  65536, {}),
    ("ram_too_small", 32, 1, {"ram": 0x200000, "reserve": 0x300000}),                          # reserve > RAM
    ("ram_too_small", 32, 1, {"ram": 0x200000, "reserve": 0x200000 - DEVICE_STACKS + 4}),     # 4 bytes short
    # Host CPU launches through the LAUNCH opcode (0x5B) -- phase C.
    ("host_launch",       1,   3, {}),
    ("host_launch",       40,  5, {}),
    ("host_launch",       128, 9, {}),
    ("host_two_launches", 40,  5, {"n": 190}),
    ("host_bad_launch",   0,   0, {}),
    ("host_no_gpu",       40,  5, {}),
    # Instructions the GPU can't run: fault, stop the block, abandon the grid.
    *[("bad_instr", 40, 5, {"bad": bad}) for bad in BAD_INSTRUCTIONS],
    ("fence_ok",          40,  5, {}),
    # ... and the CPU halts when a launch comes back failed.
    ("host_fault",        40,  5, {"kernel": "bad_instr"}),
    ("host_fault",        32,  3, {"kernel": "runaway"}),
]
TESTS = {"identity": test_identity, "stack": test_stack, "vec_add": test_vec_add,
         "divergent": test_divergent, "runaway": test_runaway,
         "bad_launch": test_bad_launch, "ram_too_small": test_ram_too_small,
         "host_launch": test_host_launch, "host_two_launches": test_host_two_launches,
         "host_bad_launch": test_host_bad_launch, "host_no_gpu": test_host_no_gpu,
         "bad_instr": test_bad_instr, "fence_ok": test_fence_ok, "host_fault": test_host_fault}


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
