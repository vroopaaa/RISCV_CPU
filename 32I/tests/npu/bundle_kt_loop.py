#!/usr/bin/env python3
"""
VLIW-style bundler for the NPU matmul Kt-loop idiom.

At -O2, tests/npu/c_tests/matmul_template_2.c's innermost Kt-loop compiles
to a fixed 6-instruction body: LOAD_A, LOAD_B, TRIG (the MAC-trigger store),
INC_A, INC_B (tile-pointer increments), BR (the loop's own back-edge). This
script finds that exact idiom in a compiled binary, reorders it into two
VLIW-style bundles -- Bundle 1 = {LOAD_A, LOAD_B, INC_A, INC_B}, Bundle 2 =
{TRIG, BR} -- and patches a copy of the flat binary so the emulator (see the
bundling block in CPU.h/CPU.cpp) executes each bundle in one cycle instead
of one cycle per instruction.

This is NOT a general instruction scheduler. It recognizes exactly one
hand-verified idiom via structural + register + constant-value cross-checks,
and leaves anything that doesn't match untouched. TRIG must end up in a
bundle that starts strictly after Bundle 1 -- it operates on the NPU's
internal resident-tile state, a hazard invisible to register-level analysis
-- which is why this script, not the CPU, is the one place that's allowed to
know this idiom's shape at all.

run_npu_tests.py always compiles at -O0 (its RISCV_FLAGS hardcodes this),
and at -O0 the Kt-loop is real function calls (jal npu_load_a / ...), not
this inlined idiom -- so this script does its own -O2 compile, reusing
run_npu_tests.py's matrix-generation/template-filling code as a library
rather than depending on its .bin output.
"""

import argparse
import os
import re
import struct
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # tests/npu -> tests -> 32I
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

sys.path.insert(0, SCRIPT_DIR)
from run_npu_tests import generate_matrix, render_test_c, TEMPLATE_PATHS  # noqa: E402

START_S = os.path.join(PROJECT_ROOT, "tests", "python", "start.s")
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")
RISCV_CC = "riscv64-unknown-elf-gcc"
RISCV_OBJCOPY = "riscv64-unknown-elf-objcopy"
RISCV_OBJDUMP = "riscv64-unknown-elf-objdump"

RAM_SIZE = 4 * 1024 * 1024  # link.ld: RAM LENGTH = 4M -- code/data addresses live below this;
                             # NPU MMIO (0x80000000+) is always far above it, which is what lets
                             # the relocation pass tell "this constant is a .rodata pointer that
                             # moved" apart from "this constant is a peripheral register address
                             # that must never move".
NPU_BASE = 0x80000000
MAC_ADDR = NPU_BASE + 0x10

BUNDLE_MARKER_OPCODE = 0x2B
NPU_OPCODE = 0x0B
WINDOW_LEN = 6


# ---------------------------------------------------------------------------
# Build the -O2 candidate binary
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def build_candidate(big_dim, seed, low, high, name):
    """Fills matmul_template_2.c (reusing run_npu_tests.py's matrix/template
    helpers) and cross-compiles it at -O2 -- unlike run_npu_tests.py itself,
    which always builds at -O0 and therefore never produces the inlined
    idiom this script looks for. Returns (elf_path, bin_path)."""
    import random
    os.makedirs(BUILD_DIR, exist_ok=True)
    rng = random.Random(seed)
    A = generate_matrix(big_dim, low, high, rng)
    B = generate_matrix(big_dim, low, high, rng)

    c_path = os.path.join(BUILD_DIR, name + ".c")
    render_test_c(big_dim, A, B, c_path, TEMPLATE_PATHS[2], 2)

    elf_path = os.path.join(BUILD_DIR, name + "_O2.elf")
    bin_path = os.path.join(BUILD_DIR, name + "_O2.bin")
    run([
        RISCV_CC, "-march=rv32im", "-mabi=ilp32",
        "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2",
        "-T", LINK_LD, "-o", elf_path, START_S, c_path,
    ])
    run([RISCV_OBJCOPY, "-O", "binary", elf_path, bin_path])
    return elf_path, bin_path


# ---------------------------------------------------------------------------
# Disassemble + decode
# ---------------------------------------------------------------------------

def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


class Insn:
    """One decoded instruction word -- fields extracted the same way
    CPU::decode_one does, so this matches what the emulator will actually
    see, not what objdump's mnemonic text happens to print."""

    __slots__ = ("addr", "word", "opcode", "rd", "rs1", "rs2", "funct3", "funct7",
                 "imm_i", "imm_s", "imm_b", "imm_j", "imm_u")

    def __init__(self, addr, word):
        self.addr = addr
        self.word = word
        self.opcode = word & 0x7F
        self.rd = (word >> 7) & 0x1F
        self.funct3 = (word >> 12) & 0x7
        self.rs1 = (word >> 15) & 0x1F
        self.rs2 = (word >> 20) & 0x1F
        self.funct7 = (word >> 25) & 0x7F
        self.imm_i = sign_extend(word >> 20, 12)
        self.imm_s = sign_extend((((word >> 25) & 0x7F) << 5) | ((word >> 7) & 0x1F), 12)
        self.imm_b = sign_extend(
            (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11) |
            (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1), 13)
        self.imm_j = sign_extend(
            (((word >> 31) & 1) << 20) | (((word >> 12) & 0xFF) << 12) |
            (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1), 21)
        self.imm_u = word & 0xFFFFF000


_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s")


def disassemble(elf_path):
    text = run([RISCV_OBJDUMP, "-d", elf_path]).stdout
    insns = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        insns.append(Insn(int(m.group(1), 16), int(m.group(2), 16)))
    # Every word-list-index-as-address trick below depends on .text being a
    # gapless run of 4-byte instructions starting at 0 -- true for this
    # project's link.ld (ORIGIN = 0) and this toolchain (no jump tables /
    # data-in-.text in any of these generated tests).
    for i, insn in enumerate(insns):
        assert insn.addr == i * 4, (
            f"gap or misalignment in .text at index {i}: expected addr "
            f"0x{i * 4:x}, got 0x{insn.addr:x} -- word-list addressing "
            "assumption below doesn't hold for this binary"
        )
    return insns


# ---------------------------------------------------------------------------
# A simple linear constant tracker: lui rX,imm_hi followed later by
# addi rX,rX,imm_lo (same register, nothing else writing rX in between) is
# this compiler's standard idiom for materializing an absolute 32-bit
# constant. Used to (a) verify TRIG really targets MAC_ADDR by its VALUE,
# not just its position, and (b) find the &A[...]/&B[...] address loads the
# relocation pass needs to adjust.
# ---------------------------------------------------------------------------

_WRITES_RD = {0x13, 0x33, 0x37, 0x17, 0x03, 0x6F, 0x67}


class BundlingUnsupported(Exception):
    """Raised when a binary uses an absolute-address idiom this script's
    relocation pass doesn't recognize -- e.g. a `lui` whose result feeds an
    address through something other than a same-register completing `addi`
    or a relaxed bare `li` (observed at 512x512: a `lui` combined via
    `sub`/`add` instead of completed by an `addi`). Rather than guess and
    risk silently mis-relocating (or failing to relocate) a real address,
    bundling is refused for binaries whose addressing this script can't
    fully account for. Verified/supported range: BIG_DIM 64 through 256
    (TILES 4 through 16) -- see the sweep results in the project notes."""


def track_constants(insns):
    """Returns (constants, unresolved_luis).

    `constants` is {addr: (rd, value, kind)} for every simple constant-
    materializing instruction: kind 'lui_addi' (lui rX,hi + addi rX,rX,lo)
    or kind 'li' (a bare addi rd,x0,imm -- RISC-V linker relaxation
    collapses a lui+addi pair down to this single-instruction form whenever
    the final linked address's upper 20 bits happen to be zero, which is
    common for whichever of A/B lands first in .rodata, right after a small
    .text section). These are candidates only -- a bare small immediate is
    indistinguishable from an ordinary integer constant on its own, so
    callers decide which values are actually relocation-eligible (see
    bundle_match's use of the ELF's real symbol table).

    `unresolved_luis` is the address of every `lui` whose value we lost
    track of -- reassigned to something other than a same-register
    completing `addi`, or still pending at the end of the function. Any
    such `lui` means this program uses an addressing idiom this tracker
    doesn't understand, so its value might be an address needing
    relocation that we'd otherwise silently miss (or corrupt).
    """
    pending_lui = {}   # register -> (most recent lui's imm_u, that lui's address)
    constants = {}      # address of the completing/bare addi -> (register, value, kind)
    unresolved_luis = []
    for insn in insns:
        if insn.opcode == 0x37:  # LUI
            pending_lui[insn.rd] = (insn.imm_u, insn.addr)
            continue
        if insn.opcode == 0x13 and insn.funct3 == 0x0:  # ADDI (incl. the li pseudo-op)
            if insn.rs1 == 0:
                constants[insn.addr] = (insn.rd, insn.imm_i & 0xFFFFFFFF, "li")
                pending_lui.pop(insn.rd, None)
                continue
            if insn.rd == insn.rs1 and insn.rs1 in pending_lui:
                imm_u, _lui_addr = pending_lui[insn.rs1]
                value = (imm_u + insn.imm_i) & 0xFFFFFFFF
                constants[insn.addr] = (insn.rd, value, "lui_addi")
                del pending_lui[insn.rs1]
                continue
        if insn.opcode in _WRITES_RD and insn.rd != 0:
            if insn.rd in pending_lui:
                unresolved_luis.append(pending_lui[insn.rd][1])
            pending_lui.pop(insn.rd, None)  # rd reassigned some other way -- no longer pending
    unresolved_luis.extend(addr for _imm_u, addr in pending_lui.values())  # still pending at EOF
    return constants, unresolved_luis


_NM_LINE_RE = re.compile(r"^([0-9a-fA-F]+)\s+\S\s+(\S+)$")


def symbol_addresses(elf_path):
    """Ground truth for what counts as a real A/B address, from the linked
    binary's own symbol table -- deliberately not a numeric-range guess
    (e.g. "any small-ish constant"), since an ordinary integer like BIG_DIM
    can otherwise coincidentally fall in a plausible-looking address range."""
    result = run(["riscv64-unknown-elf-nm", elf_path])
    addrs = {}
    for line in result.stdout.splitlines():
        m = _NM_LINE_RE.match(line.strip())
        if m:
            addrs[m.group(2)] = int(m.group(1), 16)
    return addrs


def register_value_before(addr_limit, reg, constants):
    best = None
    for addr, (r, value, _kind) in constants.items():
        if r == reg and addr < addr_limit and (best is None or addr > best[0]):
            best = (addr, value)
    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def find_bundle_windows(insns):
    """Scans every contiguous 6-instruction window; returns a list of match
    dicts (w0..w5 = the Insn objects). A window failing any check is simply
    skipped -- correctness over coverage."""
    constants, _unresolved = track_constants(insns)

    branch_targets = set()
    for insn in insns:
        if insn.opcode == 0x63:
            branch_targets.add(insn.addr + insn.imm_b)
        elif insn.opcode == 0x6F:
            branch_targets.add(insn.addr + insn.imm_j)

    matches = []
    for idx in range(len(insns) - WINDOW_LEN + 1):
        w0, w1, w2, w3, w4, w5 = insns[idx:idx + WINDOW_LEN]

        if not (w0.opcode == NPU_OPCODE and w0.funct3 == 0x0):          # LOAD_A
            continue
        if not (w1.opcode == NPU_OPCODE and w1.funct3 == 0x1):          # LOAD_B
            continue
        if not (w2.opcode == 0x23 and w2.funct3 == 0x2):                # sw (word store) -- TRIG
            continue
        if not (w3.opcode == 0x13 and w3.funct3 == 0x0
                and w3.rd == w3.rs1 == w0.rs1):                          # addi Ra,Ra,imm -- INC_A
            continue
        if not (w4.opcode == 0x33 and w4.funct3 == 0x0 and w4.funct7 == 0x0
                and w4.rd == w4.rs1 == w1.rs1):                          # add Rb,Rb,Rc -- INC_B
            continue
        if not (w5.opcode == 0x63 and w5.funct3 == 0x1):                # bne -- BR
            continue
        if w5.addr + w5.imm_b != w0.addr:                                # tight self-loop
            continue
        if w5.rs1 != w3.rd:                                              # BR compares the just-incremented Ra
            continue

        rv = register_value_before(w2.addr, w2.rs1, constants)
        if rv is None or (rv + w2.imm_s) != MAC_ADDR:                    # TRIG really targets MAC_ADDR
            continue

        interior = {w1.addr, w2.addr, w3.addr, w4.addr, w5.addr}
        if branch_targets & interior:                                    # nothing branches into the interior
            continue

        matches.append({"w0": w0, "w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5})
    return matches


# ---------------------------------------------------------------------------
# Re-encoding helpers (inverse of Insn's field extraction)
# ---------------------------------------------------------------------------

def encode_b(rs1, rs2, funct3, imm):
    imm &= 0x1FFF
    bit12 = (imm >> 12) & 1
    bit11 = (imm >> 11) & 1
    bits10_5 = (imm >> 5) & 0x3F
    bits4_1 = (imm >> 1) & 0xF
    return ((bit12 << 31) | (bits10_5 << 25) | (rs2 << 20) | (rs1 << 15)
             | (funct3 << 12) | (bits4_1 << 8) | (bit11 << 7) | 0x63)


def encode_i(rs1, funct3, rd, imm, opcode):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def pack_words(words):
    return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)


# ---------------------------------------------------------------------------
# Reorder + insert markers + relocate
# ---------------------------------------------------------------------------

def bundle_match(insns, data, match, elf_path, big_dim):
    """Returns the fully patched flat binary (bytes) for a single match.
    `insns` and `data` are both for the UNPATCHED binary. `elf_path` and
    `big_dim` are needed to look up A/B's real linked addresses (ground
    truth for the relocation pass below)."""
    w0, w1, w2, w3, w4, w5 = (match[k] for k in ("w0", "w1", "w2", "w3", "w4", "w5"))
    window_start = w0.addr
    window_end = w5.addr + 4
    delta = 8  # window grows from 6 words (24B) to 8 words (32B): two extra marker words

    text_size = len(insns) * 4
    assert w5.addr + 4 <= text_size

    words = [insn.word for insn in insns]

    # --- 1. Reorder + insert markers within the window ---------------------
    marker1 = (4 << 25) | BUNDLE_MARKER_OPCODE  # Bundle 1: K=4 real instructions follow
    marker2 = (2 << 25) | BUNDLE_MARKER_OPCODE  # Bundle 2: K=2 real instructions follow

    new_br_addr = window_start + 7 * 4  # BR's new position: last of the 8 new words
    new_br_word = encode_b(w5.rs1, w5.rs2, w5.funct3, window_start - new_br_addr)

    window_start_idx = window_start // 4
    window_end_idx = window_end // 4
    new_window_words = [marker1, w0.word, w1.word, w3.word, w4.word,
                         marker2, w2.word, new_br_word]
    words = words[:window_start_idx] + new_window_words + words[window_end_idx:]

    # --- 2. Relocate branches/jumps whose source and target straddle the
    #        insertion point (same-side pairs are self-correcting) ---------
    for insn in insns:
        if window_start <= insn.addr < window_end:
            continue  # already rebuilt above
        if insn.opcode == 0x63:
            old_target = insn.addr + insn.imm_b
        elif insn.opcode == 0x6F:
            old_target = insn.addr + insn.imm_j
        else:
            continue
        src_after = insn.addr >= window_end
        tgt_after = old_target >= window_end
        if src_after == tgt_after:
            continue
        new_addr = insn.addr + (delta if src_after else 0)
        new_target = old_target + (delta if tgt_after else 0)
        new_imm = new_target - new_addr
        if insn.opcode == 0x63:
            new_word = encode_b(insn.rs1, insn.rs2, insn.funct3, new_imm)
        else:
            new_word = (((new_imm >> 20) & 1) << 31) | (((new_imm >> 1) & 0x3FF) << 21) \
                       | (((new_imm >> 11) & 1) << 20) | (((new_imm >> 12) & 0xFF) << 12) \
                       | (insn.rd << 7) | 0x6F
        words[new_addr // 4] = new_word

    # --- 3. Relocate absolute-address constants (lui+addi pairs, or a
    #        relaxed bare li -- see track_constants) whose value lands in
    #        &A[...]/&B[...]'s real address range (from the ELF's own
    #        symbol table, not a numeric-range guess -- an ordinary integer
    #        constant like BIG_DIM can otherwise coincidentally look like a
    #        plausible address) at or past the insertion point. NPU MMIO
    #        constants (0x80000000+) are never anywhere near this range. --
    #
    # Safety net: if this program used ANY addressing idiom we don't
    # recognize (a lui never completed by a matching addi/li -- e.g. one
    # combined via sub/add instead, seen at 512x512), we can't be confident
    # we've found every constant that needs relocating. Refuse to bundle
    # rather than risk silently emitting a wrong result.
    constants, unresolved_luis = track_constants(insns)
    if unresolved_luis:
        raise BundlingUnsupported(
            f"{len(unresolved_luis)} lui instruction(s) at "
            f"{[hex(a) for a in unresolved_luis]} use an addressing idiom "
            "this relocation pass doesn't recognize (not a same-register "
            "lui+addi pair or a relaxed bare li) -- refusing to bundle "
            "rather than risk an unrelocated address."
        )

    syms = symbol_addresses(elf_path)
    size_bytes = big_dim * big_dim * 4
    ranges = [(syms[name], syms[name] + size_bytes) for name in ("A", "B") if name in syms]
    exact = {syms[name] for name in ("A", "B") if name in syms}

    def relocation_eligible(value, kind):
        if kind == "li":
            return value in exact  # bare li only ever observed as an exact base, not base+offset
        return any(lo <= value < hi for lo, hi in ranges)

    for addi_addr, (reg, value, kind) in constants.items():
        if not relocation_eligible(value, kind) or value < window_end:
            continue
        addi_insn = insns[addi_addr // 4]
        new_imm = addi_insn.imm_i + delta
        if not (-2048 <= new_imm <= 2047):
            raise RuntimeError(
                f"relocation overflow: addi at 0x{addi_addr:x} immediate "
                f"{addi_insn.imm_i} + {delta} doesn't fit in 12 bits -- "
                "would need adjusting the paired lui too, not handled"
            )
        new_addr = addi_addr + (delta if addi_addr >= window_end else 0)
        words[new_addr // 4] = encode_i(addi_insn.rs1, addi_insn.funct3, addi_insn.rd,
                                          new_imm, addi_insn.opcode)

    return pack_words(words) + data[text_size:]


def bundle_binary(elf_path, bin_path, big_dim):
    """Finds Kt-loop idiom instances in elf_path and returns the patched
    flat binary bytes (bin_path's content, with exactly one window bundled).
    Raises if more than one match is found -- this template produces exactly
    one static Kt-loop body, and applying multiple insertions' relocations
    together isn't exercised/verified by this script. Raises
    BundlingUnsupported (see track_constants) if the binary's addressing
    isn't one this script's relocation pass can fully account for --
    callers that want to fall back to the unbundled binary in that case
    should catch it explicitly, rather than this function silently doing so
    (verified/supported range: BIG_DIM 64-256)."""
    insns = disassemble(elf_path)
    with open(bin_path, "rb") as f:
        data = f.read()

    matches = find_bundle_windows(insns)
    if not matches:
        return data, 0
    if len(matches) > 1:
        raise RuntimeError(
            f"found {len(matches)} Kt-loop matches; this script only handles "
            "exactly one (matmul_template_2.c has a single static Kt-loop body)"
        )
    patched = bundle_match(insns, data, matches[0], elf_path, big_dim)
    return patched, 1


# ---------------------------------------------------------------------------
# CLI (manual/diagnostic use -- run_bundle_tests.py is the real driver)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bundle the NPU matmul Kt-loop idiom (-O2 only).")
    parser.add_argument("--size", type=int, default=256, help="Square matrix size (multiple of 16)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--low", type=int, default=-10)
    parser.add_argument("--high", type=int, default=10)
    args = parser.parse_args()

    name = f"matmul_{args.size}_bundle"
    elf_path, bin_path = build_candidate(args.size, args.seed, args.low, args.high, name)
    try:
        patched, n = bundle_binary(elf_path, bin_path, args.size)
    except BundlingUnsupported as e:
        print(f"Refusing to bundle: {e}")
        sys.exit(1)

    if n == 0:
        print("No Kt-loop bundle candidates found.")
        sys.exit(1)

    out_path = os.path.join(BUILD_DIR, name + ".bundled.bin")
    with open(out_path, "wb") as f:
        f.write(patched)
    print(f"Bundled {n} window(s). Original {os.path.getsize(bin_path)} bytes -> "
          f"{len(patched)} bytes. Wrote {out_path}")


if __name__ == "__main__":
    main()
