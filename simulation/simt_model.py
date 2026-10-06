"""
Python prototype of the SIMT execution model: one warp, T lanes, one shared
memory, lockstep instruction broadcast with per-lane masking.

This is deliberately NOT RISC-V -- no encoding, no decode step. A kernel is
just a list of symbolic ops (see kernels.py). The point is to pin down the
execution semantics (warp/lane/mask/shared-memory behavior) before porting
them into 32I/include/SIMT.h + 32I/src/SIMT.cpp.

Maps onto the C++ side like this:
  Warp.regs[lane][reg]      <-> SIMTCore::regfile[lane][warp][reg]
  Warp.pc / Warp.tmask      <-> per-warp pc/tmask fields in SIMTCore
  SIMTCore.shared_mem       <-> the Memory* every lane's LOAD/STORE goes through
  SIMTCore.step()           <-> SIMTCore::issue(warp) in C++
"""

from dataclasses import dataclass, field


THREADS_PER_WARP = 4  # T -- matches 32I/include/SIMT.h's SIMTCore::THREADS_PER_WARP
NUM_REGS = 8           # small register file per lane; grow if a kernel needs more
IPDOM_DEPTH = 8        # max nested SPLITs before reconverging; matches the reverted
                        # C++ draft's IpdomEntry ipdom[IPDOM_DEPTH] -- not enforced here
                        # (plain list), just documenting the eventual fixed-size bound.

# Ops that are WARP-level, not lane-level: they read/write warp.tmask, warp.pc, or
# warp.ipdom_stack directly instead of a single lane's registers, so step() must
# call them ONCE PER CYCLE (handler sees the whole warp), never once per active
# lane like TID/LOADI/LOAD/STORE/ADD/MUL/ADDI are. See _op_split/_op_join/_op_branch/
# _op_halt docstrings below for what each one actually needs from the whole warp.
WARP_LEVEL_OPS = {"BRANCH", "SPLIT", "JOIN", "HALT"}


def _to_signed32(v: int) -> int:
    """Reinterpret a stored (always-masked-to-32-bit) register value as
    signed, the way every handler below needs to before a signed op."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


@dataclass
class Warp:
    """One warp: a single PC and active-thread mask shared by all lanes,
    plus each lane's own private registers (register file is banked by
    lane, same as the C++ regfile -- lane t never touches lane u's row)."""

    pc: int = 0
    tmask: int = (1 << THREADS_PER_WARP) - 1  # all lanes active by default
    regs: list = field(
        default_factory=lambda: [[0] * NUM_REGS for _ in range(THREADS_PER_WARP)]
    )
    halted: bool = False
    # Reconvergence stack: each entry is (old_mask, reconv_pc), pushed by SPLIT,
    # popped by JOIN. "old_mask" is the FULL pre-branch mask (not the untaken
    # half) -- SPLIT narrows tmask to just the taken lanes; JOIN restores it.
    # This is a predication model, not the full two-path hardware SIMT stack:
    # if/else is two separate SPLIT/body/JOIN blocks in program order (see
    # kernels.py's manufactured-branch kernel), not one SPLIT branching to two
    # different PCs. A plain list stands in for the fixed-depth array
    # (IPDOM_DEPTH) the C++ port will need.
    ipdom_stack: list = field(default_factory=list)

    def lane_active(self, lane: int) -> bool:
        return bool(self.tmask & (1 << lane))


class SIMTCore:
    """Single warp, T lanes, one shared memory. Fetches one op per cycle at
    warp.pc, broadcasts it to every active lane, advances pc once for the
    whole warp (true lockstep -- no per-lane PCs here, same constraint the
    C++ SIMTCore has: one Warp::pc for all its threads)."""

    def __init__(self, shared_mem_size: int):
        self.shared_mem = [0] * shared_mem_size
        self.warp = Warp()
        self.cycle = 0
        self.trace = []  # list of strings, one per cycle, if you want a log

        # Dispatch table: mnemonic -> handler(self, lane, *args) -> None.
        # Each handler reads/writes self.warp.regs[lane] and/or
        # self.shared_mem for just that one lane. TODO: fill these in.
        self._handlers = {
            "TID": self._op_tid,
            "LOADI": self._op_loadi,
            "LOAD": self._op_load,
            "STORE": self._op_store,
            "ADD": self._op_add,
            "MUL": self._op_mul,
            "ADDI": self._op_addi,
            "SLT": self._op_slt,
            "DIV": self._op_div,
            # ---- warp-level ops: see WARP_LEVEL_OPS -- step() must call these
            # ONCE PER CYCLE, not once per active lane like everything above ----
            "BRANCH": self._op_branch,
            "SPLIT": self._op_split,
            "JOIN": self._op_join,
            "HALT": self._op_halt,
        }

    def load_kernel(self, kernel: list, start_pc: int = 0, tmask: int = None):
        """Point the warp at a kernel (see kernels.py) and reset its state."""
        self.kernel = kernel
        self.warp = Warp(pc=start_pc)
        if tmask is not None:
            self.warp.tmask = tmask

    def run(self, max_cycles: int = 1000) -> int:
        """Runs step() until the warp halts or max_cycles elapse. Returns
        the number of cycles actually run."""
        while (not self.warp.halted) and (self.cycle < max_cycles):
            self.step()
        return self.cycle

    def step(self) -> None:
        """One cycle: fetch kernel[warp.pc], dispatch it, advance pc and
        cycle. Lane-level ops are broadcast once per active lane (passed
        `lane` plus the op's own args). Warp-level ops (see WARP_LEVEL_OPS)
        are called exactly once, with no `lane` argument, and may return a
        next-pc override -- only BRANCH actually uses that (SPLIT/JOIN/HALT
        always fall through to pc+1, same default as every lane op)."""
        instr, *args = self.kernel[self.warp.pc]
        handler = self._handlers[instr]

        next_pc = self.warp.pc + 1  # default: straight-line fall-through
        if instr in WARP_LEVEL_OPS:
            override = handler(*args)
            if override is not None:
                next_pc = override
        else:
            for lane in range(THREADS_PER_WARP):
                if self.warp.lane_active(lane):
                    handler(lane, *args)

        self.warp.pc = next_pc
        self.cycle += 1

    # ---- op handlers ---------------------------------------------------
    # Each one operates on a SINGLE lane. step() is responsible for calling
    # the right handler once per active lane -- the handlers themselves
    # must never look at other lanes' registers (that's the whole point of
    # per-lane register banking).

    def _op_tid(self, lane: int) -> None:
        """Write this lane's own index into a fixed "idx register" --
        stands in for the real TID custom-1 instruction. TODO."""
        self.warp.regs[lane][0] = lane  # just use reg 0 for now

    def _op_loadi(self, lane: int, reg: int, value: int) -> None:
        self.warp.regs[lane][reg] = value

    def _op_load(self, lane: int, dst_reg: int, addr_reg: int) -> None:
        """regs[lane][dst_reg] = shared_mem[regs[lane][addr_reg]]"""
        self.warp.regs[lane][dst_reg] = self.shared_mem[self.warp.regs[lane][addr_reg]]

    def _op_store(self, lane: int, addr_reg: int, src_reg: int) -> None:
        """shared_mem[regs[lane][addr_reg]] = regs[lane][src_reg]"""
        self.shared_mem[self.warp.regs[lane][addr_reg]] = self.warp.regs[lane][src_reg]

    def _op_add(self, lane: int, dst_reg: int, src1_reg: int, src2_reg: int) -> None:
        self.warp.regs[lane][dst_reg] = (self.warp.regs[lane][src1_reg] + self.warp.regs[lane][src2_reg]) & 0xFFFFFFFF  # 32-bit wraparound

    def _op_addi(self, lane: int, dst_reg: int, src_reg: int, imm: int) -> None:
        """Add immediate: regs[lane][dst_reg] = regs[lane][src_reg] + imm"""
        self.warp.regs[lane][dst_reg] = (self.warp.regs[lane][src_reg] + imm) & 0xFFFFFFFF  # 32-bit wraparound

    def _op_mul(self, lane: int, dst_reg: int, src1_reg: int, src2_reg: int) -> None:
        self.warp.regs[lane][dst_reg] = (self.warp.regs[lane][src1_reg] * self.warp.regs[lane][src2_reg]) & 0xFFFFFFFF  # 32-bit wraparound

    def _op_slt(self, lane: int, dst_reg: int, src1_reg: int, src2_reg: int) -> None:
        """Set-less-than: regs[lane][dst_reg] = 1 if regs[lane][src1_reg] <
        regs[lane][src2_reg] else 0. Per-lane boolean, the same shape as real
        RV32I's `slt` -- this is how a kernel produces the PER-LANE predicate
        that SPLIT later reads (e.g. SLT computing `idx < n` for each lane
        before a SPLIT/JOIN-guarded block). TODO."""
        self.warp.regs[lane][dst_reg] = 1 if self.warp.regs[lane][src1_reg] < self.warp.regs[lane][src2_reg] else 0

    def _op_div(self, lane: int, dst_reg: int, src1_reg: int, src2_reg: int) -> None:
        """Signed division, truncating toward zero -- matches RV32IM's DIV
        (and CPU.cpp's case 0x4: `(int32_t)rs1 / (int32_t)rs2`). Division by
        zero returns -1 (0xFFFFFFFF), same no-trap convention as the rest of
        this codebase; there's no INT_MIN/-1 overflow case to special-case
        here the way CPU.cpp does, since Python ints don't overflow."""
        a = _to_signed32(self.warp.regs[lane][src1_reg])
        b = _to_signed32(self.warp.regs[lane][src2_reg])
        if b == 0:
            result = -1
        else:
            q = abs(a) // abs(b)
            result = -q if (a < 0) != (b < 0) else q
        self.warp.regs[lane][dst_reg] = result & 0xFFFFFFFF

    # ---- warp-level ops ---------------------------------------------------
    # Unlike every handler above, these do NOT get called once per active
    # lane -- step() must call each of these exactly ONCE per cycle (see
    # WARP_LEVEL_OPS), with access to the whole warp, not a single lane.

    def _op_split(self, old_pred_reg: int, reconv_pc: int) -> None:
        """Divergent-branch entry: narrows the active mask to just the lanes
        whose predicate is true, and remembers how to get everyone back.

        Algorithm (see simulation/README.md's SIMT model notes):
          old_mask = warp.tmask
          taken_mask = 0
          for lane in range(THREADS_PER_WARP):
              if warp.lane_active(lane) and warp.regs[lane][pred_reg] != 0:
                  taken_mask |= (1 << lane)
          warp.ipdom_stack.append((old_mask, reconv_pc))
          warp.tmask = taken_mask
          # pc advances to pc+1 as normal -- SPLIT never jumps itself.

        Note the signature here takes no `lane` argument (unlike the lane ops
        above) -- it needs ALL active lanes' pred_reg values at once to build
        taken_mask, which is exactly why this has to be a warp-level op.
        """
        old_mask = self.warp.tmask
        taken_mask = 0
        for lane in range(THREADS_PER_WARP):
            if self.warp.lane_active(lane):
                if self.warp.regs[lane][old_pred_reg] != 0:
                    taken_mask |= (1 << lane)
        self.warp.ipdom_stack.append((old_mask, reconv_pc))
        self.warp.tmask = taken_mask

    def _op_join(self) -> None:
        """Reconvergence: pop the IPDOM stack and restore the mask SPLIT
        narrowed away from.

        Algorithm:
          old_mask, reconv_pc = warp.ipdom_stack.pop()
          warp.tmask = old_mask
          # pc advances to pc+1 as normal.

        reconv_pc is popped but unused by this predication model (see the
        Warp.ipdom_stack docstring) -- it's carried along mainly because the
        eventual C++ IpdomEntry has the field, and because a future nested-
        divergence sanity check (e.g. asserting pc == reconv_pc at JOIN time)
        would want it. TODO.
        """
        old_mask, reconv_pc = self.warp.ipdom_stack.pop()
        self.warp.tmask = old_mask

    def _op_branch(self, cond_reg: int, target_pc: int):
        """Uniform branch: every ACTIVE lane must agree on cond_reg's
        truthiness, or this raises -- a data-dependent condition that can
        differ per lane belongs behind SPLIT/JOIN, not BRANCH (see
        simulation/README.md). Returns the pc to jump to if taken, or None
        if not taken -- step() falls back to pc+1 when this returns None,
        same contract as _op_split/_op_join (which never override pc)."""
        active_values = [
            bool(self.warp.regs[lane][cond_reg])
            for lane in range(THREADS_PER_WARP)
            if self.warp.lane_active(lane)
        ]
        distinct = set(active_values)
        if len(distinct) > 1:
            raise ValueError(
                f"BRANCH divergence: active lanes disagree on cond_reg {cond_reg} "
                f"(mask={self.warp.tmask:0{THREADS_PER_WARP}b}) -- use SPLIT/JOIN "
                "for a condition that can differ per lane"
            )
        taken = distinct.pop() if distinct else False  # no active lanes: outcome is moot
        return target_pc if taken else None

    def _op_halt(self) -> None:
        """Retire the warp. Warp-level: reaching a HALT instruction's pc
        means every currently-active lane got there in the same cycle (true
        lockstep -- there's no way for only some active lanes to "reach" an
        instruction, they all execute whatever's at warp.pc together), so
        there's no agreement check needed the way BRANCH has one."""
        self.warp.halted = True