# SIMT layer progress log

A chronological record of the SIMT work on `32I`: what was built, what
design questions came up and how they were resolved, and what's still
open. `docs/CUDA/plan.md` is the up-to-date design reference (current
rules only); this file is the story of how it got there.

## 1. Python prototype first

Before touching C++, the execution model was built and validated as a
standalone prototype in `simulation/` -- deliberately not RISC-V (no
encoding, no decode step): a kernel is a list of symbolic ops, interpreted
by `simt_model.py`'s `SIMTCore`/`Warp`. Reasoning: validate warp/lane/
mask/shared-memory *semantics* fast, without a compiler or instruction
encoding in the way, then port known-correct semantics to C++ rather than
design and encode at the same time.

Built up incrementally:
- `Warp` (pc, tmask, per-lane `regs`) + a lane-broadcast `step()`/`run()`.
- Op handlers: `TID`, `LOADI`, `LOAD`/`STORE` (through one shared
  `shared_mem` list -- this model's only cross-lane communication channel,
  since registers are strictly per-lane), `ADD`/`ADDI`/`MUL`/`SLT`.
- `WARP_LEVEL_OPS` split in `step()`'s dispatch: most ops broadcast once
  per active lane, but `BRANCH`/`SPLIT`/`JOIN`/`HALT` need the whole warp
  at once (to build a mask, or because there's nothing per-lane to do) and
  are called exactly once per cycle instead.
- `_op_branch`: requires every active lane to agree, raises otherwise
  (pointing at `SPLIT`/`JOIN`) -- established the "uniform branch vs.
  divergence-capable `SPLIT`/`JOIN`" split that carried through to C++.
- `Warp.ipdom_stack` + `_op_split`/`_op_join`: the predication model (push
  the *old* mask, narrow to the predicate; `if/else` = two SPLIT/body/JOIN
  blocks in program order) -- see `plan.md` for why this shape, not the
  full two-path hardware stack.
- `DIV` added when the softmax kernel needed division (nothing else in the
  symbolic ISA had one) -- signed, truncating toward zero, divide-by-zero
  returns -1, matching `CPU.cpp`'s real `DIV`.

Kernels built and verified against independently-computed expected output
(same "recompute, diff" philosophy as `32I/tests/npu/run_npu_tests.py`):
- `SCALAR_MULTIPLY_KERNEL` -- the reduced CUDA example (N=4, 1 block, 4
  threads/block, no boundary check needed since N==T).
- `DIVERGENT_BRANCH_KERNEL` -- manufactured `if (idx<2) 100 else 200`, the
  first real `SPLIT`/`JOIN` exercise. First attempt failed: the else
  branch computed `2 < idx` (strict), wrongly excluding `idx==2` --
  classic boundary bug. Fixed by negating the first branch's already-
  computed predicate (`1 - r5`, no `NOT` op exists) instead of re-deriving
  the boundary a second time, which also structurally avoids that bug
  class.
- `SOFTMAX_KERNEL` -- fixed-point `exp()` via a lookup table (no floats/
  transcendentals anywhere in this model, same as real hardware), per-mille
  output. Needed `DIV`. Cross-lane sum works with **no barrier**, because a
  single warp is lockstep by construction (every active lane finishes
  instruction N before any lane starts N+1) -- flagged that a multi-warp
  version of this pattern *would* need a real barrier.
- `LEAKY_RELU_KERNEL` -- a more realistic divergence example than the
  manufactured one: both branches compute something genuinely different
  from the lane's own data (`DIV` vs pass-through), not just a different
  constant, which is closer to why real divergence costs cycles.
- `trace.py` -- temporary (explicitly disposable) cycle-by-cycle
  visualizer: per-cycle instruction/mask/IPDOM-depth, per-lane register
  state after the instruction executed. Has a small CLI
  (`python3 trace.py [kernel_name]`) and `_demo_*()` wrappers for each
  kernel above.

## 2. C++ port -- Phase 1 (register file)

Ported first, reviewed, and explicitly signed off before continuing (the
project's own "no further implementation without explicit go-ahead"
convention). `32I/include/SIMT.h` + `src/SIMT.cpp`, `tests/simt/
regfile_test.cpp`.

One design question resolved here: whether `decode_one`/`alu_exec`/the
`InstructionFields` struct should become `public` on `CPU` so `SIMTCore`
could call them. Decided against it -- `friend class SIMTCore;` instead,
keeping `CPU`'s public interface unchanged. (An earlier attempt had
already made these public as part of an early, premature refactor; it was
fully reverted -- confirmed byte-identical to `HEAD` -- before the `friend`
approach was built for real in Phase 2.)

## 3. C++ port -- Phase 2/3 (lane-broadcast engine + custom-1 ops)

`CPU.cpp`/`CPU.h` refactor: extracted `CPU::alu_exec` and
`CPU::branch_taken` as pure `static` functions (parameterized on `a`/`b`/
`pc` instead of reading `registers[]` directly), made `decode_one` static,
added the `friend` grant. Verified behavior-identical: NPU correctness
tests give the same cycle counts (805 scalar / 436 superscalar@4 / 33
superscalar@4+`-O2`) before and after.

`SIMTCore` gained a `Warp` struct (`pc`, `tmask`, `ipdom_stack`) and:
- `issue(w)` -- fetch+decode+dispatch for one warp. ALU/load/store
  broadcast to active lanes via `exec_lane` (calls `CPU::alu_exec`,
  reads/writes the shared `Memory*` per lane). `BRANCH`/`JAL`/`JALR` are
  warp-level, handled directly in `issue()`. Custom-1 (`0x2B`) routes to
  `exec_simt`.
- `exec_simt` -- `TMC`/`TID`/`SPLIT`/`JOIN`/`PRED`, same IPDOM semantics as
  the Python model. `WSPAWN`/`BAR` are explicit no-ops with a comment
  explaining why (inherently multi-warp, nothing to do yet without Phase
  4's scheduler).

Real R/I/S/U/J-type hand-assembly helpers added to `tests/simt/
exec_test.cpp` (`r_type`/`i_type`/`s_type`/`u_type`/`j_type` + mnemonic
wrappers; this file was later deleted -- see §5) -- no toolchain involved
yet, these build raw instruction words directly. Three kernels ported from
the Python prototype and passed on first full run after fixing the
harness/constructor plumbing:
1. Scalar multiply by 4 -- real `TID` instruction this time (not a
   software convention), confirms each lane's own `idx` register matches
   its lane number.
2. The divergent-branch kernel -- confirms `ipdom_stack` is empty after
   running (no leaked push/pop).
3. `TMC` masking -- confirms masked-off lanes' memory writes don't happen
   (sentinel-value check).

All of Phase 1/2/3's tests still pass, and the scalar/NPU baseline is
unaffected, confirmed after every step (`make run-simt-tests`, NPU
suite's bit-exact + cycle-count checks).

## 4. T bumped 4 -> 32 (1-SM/32-thread experiment)

Asked to try the Leaky ReLU kernel at 32 threads on 1 SM (matching real
NVIDIA warp size, rather than the original starting T=4). Turned out to be
a one-line change in `SIMT.h` -- the engine had already been written
defensively for T up to 32 from the start (`tmask` is a `uint32_t`,
`reset_warp`'s `(THREADS_PER_WARP >= 32) ? 0xFFFFFFFFu : ...` guard against
the shift-by-32 UB case, `TMC`'s `1ull << count` before truncating). Only
test-file assertions tied to the old T=4 register count needed updating
(`regfile_test.cpp`'s `PHYS_REGS == 512` -> `== 4096`).

Added `test_leaky_relu_32` to `exec_test.cpp` (since retired -- see §5;
the same kernel now lives in `tests/simt/c_tests/leaky_relu_32.c`): full
32-lane run, real
`DIV` (RV32IM's actual M-extension `DIV`, opcode `0x33`/funct7 `0x01`/
funct3 `0x4` -- not a symbolic stand-in the way the Python prototype
needed, since real RV32IM already has it), input spread across negative
and non-negative values (`x[i] = i - 16`), expected values computed with
C++'s native truncating-toward-zero integer division (matches RV32IM
`DIV` exactly, no helper needed the way the Python test needed one).
**All 32 lanes correct on the first run.**

## 5. Real toolchain integration -- `tests/simt/exec_test.cpp` retired

Asked to split the single `exec_test.cpp` (C++, hand-assembled instruction
words) into four separate tests, in C, following the NPU methodology --
i.e. real kernels compiled by the actual `riscv64-unknown-elf-gcc`, run
through a harness, diffed against an independently-computed expected result
in Python (`tests/npu/run_npu_tests.py`'s pattern), not a host-side C++ unit
test. `exec_test.cpp` was deleted; `tests/simt/c_tests/*.c` +
`run_simt_tests.py` + `harness.cpp` replace it.

Built, in order, verifying against real disassembly at each step rather
than assuming the compiler would cooperate:

1. `simt_isa.h` -- `.insn`-wrapped C functions for `TMC`/`WSPAWN`/`TID`/
   `BAR`/`SPLIT`/`JOIN`/`PRED`, same convention as the NPU's
   `npu_load_a`/etc. `SPLIT`'s `reconv_pc` operand is hardcoded to 0 (it's
   not functionally used by `JOIN` yet, see `plan.md`) rather than computing
   a real return address.
2. `harness.cpp` -- the SIMT-layer equivalent of `tests/basic/main.cpp`:
   loads a binary, drives `SIMTCore::issue(0)` in a loop instead of the
   scalar `CPU`, dumps a memory region via `Memory::dump_range`'s file
   overload for the Python driver to parse.
3. `scalar_multiply.c` -- first kernel, deliberately the simplest (no
   divergence). Confirmed `TID`/`TMC` encode exactly as designed by decoding
   the real compiled words by hand before even running it. **Failed on
   first run**: only lane 3's result was written, lanes 0-2 stayed 0.

   Root cause: `sp` is a register like any other (banked per lane), but
   every lane started with the *same value* (`tests/python/start.s`'s
   single shared `_stack_top`), and -- unknown going in -- even `-O0`'s
   simplest code spills locals to the stack via `sp`/`s0`. Every lane's
   spill of `idx` landed on the *same* shared-memory address; within one
   broadcast cycle the last lane processed (lane 3) overwrote the others,
   and the following `lw` read lane 3's value back into every lane. Fixed
   with `start_simt.s`: reads the real `TID` instruction first and offsets
   `sp` by `(thread_id+1) * 0x400` before calling `main()`, giving each
   lane its own private 1KB stack slice. Confirmed fixed by re-disassembling
   and re-running -- `[8,12,20,28]`, correct.

4. `divergent_branch.c` -- ported next specifically *because* it was
   expected to be the hard case (a real compiled `if`, not hand-assembled
   `SPLIT`/`JOIN`). **Failed on first run** with out-of-bounds memory
   accesses at addresses like `0xffffffe4` (i.e. a frame pointer that had
   effectively become ~0). Traced cycle-by-cycle with a throwaway debug
   driver (not checked in) rather than guessing: `simt_split`/`simt_join`
   are `static inline`, which GCC does not actually honor at `-O0` --  they
   compiled to real `jal`/`ret` calls. `SPLIT` narrows the active mask
   *while some lanes are still inside `simt_split`'s own call frame*; those
   masked-off lanes freeze with `sp`/`s0`/`ra` pointing at that callee's
   frame. When `JOIN` (itself also a real call) later reactivates them, the
   shared warp `pc` has moved on to wherever the surviving lanes got to --
   a different call-nesting depth -- so they read memory relative to a
   stale frame pointer. Fixed with `__attribute__((always_inline))` on
   every `simt_*` function, forcing the whole `SPLIT..JOIN` region to stay
   straight-line code inside the caller's own frame. Confirmed via
   disassembly (zero `jal`s to any `simt_*` function remained) and re-run --
   `[100,100,200,200,...,200]` across all 32 lanes, correct.

5. `tmc_masking.c` / `leaky_relu_32.c` -- both passed on first run once the
   two fixes above were in place. `tmc_masking` seeds a sentinel with the
   full mask active, narrows via `TMC(2)`, and confirms the untouched
   lanes still hold the sentinel. `leaky_relu_32` is the real-RV32IM-`DIV`
   version of the Python prototype's kernel, 32 lanes, mixed-sign input.

All four verified end to end (`make run-simt-tests` -> `run_simt_tests.py`),
Phase 1's regfile test and the NPU baseline unaffected.

**Why this matters beyond these four kernels:** both bugs are structural
properties of "many lanes, one shared `Memory*`, lockstep broadcast" --
not quirks of these specific kernels. *Any* future kernel with local
variables or non-inlined function calls under divergent control flow would
hit the same two failure modes. `start_simt.s` and the `always_inline`
requirement are therefore load-bearing parts of the toolchain contract now,
documented in `plan.md`, not one-off fixes.

## 6. Grid launch -- the host CPU launches kernels (`docs/CUDA/grid_launch_plan.md`)

Built in the plan's phase order, each phase gated by tests, on branch
`cuda_prototype`.

- **A -- an SM runs a whole block.** `SIMTCore` gained an `sm_id`, block
  context, the identity ops (`SIMT_IDENT`: blockIdx/blockDim/gridDim/hw_tid)
  and `run_block()`: ceil(block_dim/32) warps, the last one partly masked,
  run one after another to completion, with a per-block issue cap so a
  runaway kernel can't hang the host.
- **B -- `GridLauncher`.** Round-robin blocks over 4 SMs; device cycles =
  the busiest SM's total. Changed from the design: the **CPU owns the host
  stack size** (`CPU::HOST_STACK_RESERVE`) and passes it into the
  launcher's constructor, which carves the device stacks below it. Review
  found two real bugs, fixed test-first: RAM too small for the stacks went
  undetected (sp values outside RAM), and a timed-out block didn't stop the
  grid (65535 blocks x the issue cap = effectively a hang).
- **Testing methodology changed.** The first launcher test hand-encoded
  instruction words in C++, and an encoder bug (an `addi` immediate that
  didn't fit 12 bits) hid behind it. Replaced by hand-written **assembly**
  kernels assembled by the real toolchain (`grid_tests/*.s`, SIMT ops from
  `simt_macros.inc`) plus `run_grid_tests.py`, which computes expected
  values independently in Python -- readable for RTL designers, and the
  objdump listing doubles as a waveform reference. The hand-encoded
  `run_block_test.cpp` / `grid_launcher_test.cpp` were folded in and deleted.
- **C -- LAUNCH on the host CPU** (custom-2, `0x5B`). Review caught the
  Makefile not rebuilding objects on header changes (a stale `main.o`
  crashed `cpu_emulator` once `CPU`'s layout changed) and the tests
  ignoring stderr; both fixed. **GPU faults** were added at the user's
  request: anything an SM lane can't run used to be a silent no-op; now it
  prints an error, raises a flag that abandons the grid, and the host CPU
  checks the flag after the LAUNCH and halts.
- **D -- the C API** (`simt_launch`, identity helpers) and C test programs.
  Two compiler findings: (1) gcc -O2 merged the if/else SPLIT regions of
  `c_relu.c` into one branch -> `simt_split()` now returns its predicate
  through `"+r"` and kernels branch on that; (2) review then showed gcc can
  also move code onto the path of lanes SPLIT switched off (a default value
  computed only there: 22/64 right at -O2). That one can't be fixed in a
  header, so **SIMT programs are -O0 only for now** (`#error` under
  `__OPTIMIZE__`) -- the motivation for a real kernel compiler later.
  `tests/basic/main.cpp` (emul's harness) now always attaches a GPU, and
  `assemblyinstruction -trace` nests the GPU's work under the LAUNCH line.
- **E -- LAUNCH on the superscalar CPU.** It issues alone and the CPU
  stalls in that cycle; every host test reruns at issue widths 1/2/4/10.
  The phase-D stopgaps (superscalar halt guard, automatic scalar mode in
  emul/assemblyinstruction) were removed.
- **F -- these docs.**

## 7. Open work

- **Phase 4 (scheduler)**: round-robin, scoreboard, `LOAD_LATENCY`. Needed
  before `WSPAWN`/`BAR` mean anything, and before more than one warp is
  ever actually driven (today a harness calls `reset_warp`/`issue`
  directly on warp 0 only).
- **Phase 5 (NPU handoff)**: not built; an NPU (`0x0B`) instruction on an
  SM lane faults today (it used to be a silent no-op).
- **Multi-warp vecadd** now runs through a real launch (`c_vec_add.c`,
  `vec_add.s`: several blocks x several warps), but warps still run one
  after another to completion -- genuine interleaving needs Phase 4.
- **Per-lane stacks (1KB/lane) are unguarded** -- nothing checks for a
  stack overflow into the next lane's slice, the 64KB host stack isn't
  guarded against the device stacks either, and nothing stops a large
  `.bss` from growing into the device-stack area.
- **SIMT C programs are `-O0` only** (see `plan.md`, Toolchain point 3) --
  lifting that needs a compiler that places SPLIT/JOIN itself.
- **The host CPU still ignores unknown opcodes silently** (only the SM
  lanes fault on them so far).
- Small known gaps from review: `simt_launch` doesn't mask block/thread
  counts above 0xFFFF; `simt_wspawn`/`simt_bar` lack `"memory"` clobbers;
  LAUNCH ignores funct7 and doesn't validate the kernel address; a timed-out
  kernel is reported to the host only by halting.
- **`JALR` divergence**: leader-lane-only, no agreement check (see
  `plan.md`) -- acceptable for now, flagged as a known gap rather than a
  verified-safe design choice.
- **`SPLIT`'s `reconv_pc`**: still hardcoded to 0 in `simt_isa.h`, carried
  in the IPDOM stack entry but not functionally used by `JOIN`. A
  `JOIN`-time `assert(pc == reconv_pc)` sanity check (catching a kernel
  that places `JOIN` somewhere other than where its matching `SPLIT`
  promised) would be cheap to add later, and would need `simt_split()` to
  actually compute a real return address first (e.g. via GNU C's
  labels-as-values extension).
- **Per-SM shared scratchpad**: still just the flat `Memory*` (same "shared
  memory" the scalar/NPU paths use) -- a smaller, faster, genuinely
  per-SM-local scratchpad was explicitly deferred to after this lands.
