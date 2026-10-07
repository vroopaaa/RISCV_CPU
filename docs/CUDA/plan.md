# SIMT layer -- design reference

A SIMT (warp/thread) execution layer for `32I`, built alongside the existing
scalar/superscalar `CPU` and the NPU -- not inside either. Goal: run simple
data-parallel kernels (vector-add, softmax, Leaky ReLU, ...) the way a tiny
CUDA-core array would, reusing the unmodified `riscv64-unknown-elf-gcc`
toolchain (no compiler/assembler changes -- new instructions are emitted via
`.insn` inline asm, same convention the NPU's custom-0 opcode already uses).
This is the up-to-date design (current rules only); `docs/CUDA/progress.md`
is the chronological story of how it got here and what's still open. A
program on the host CPU launches kernels onto several SMs itself -- see
"Grid launch" below (full design: `docs/CUDA/grid_launch_plan.md`).

## Starting config

- `THREADS_PER_WARP` (T) = **32** (bumped from an initial 4 once the lane
  engine was proven -- see progress log; matches real NVIDIA warp size).
- `WARPS_RESIDENT` (W) = 4.
- All hardcoded in `32I/include/SIMT.h` -- not build-configurable yet (same
  posture as the rest of `32I`: prove it first, generalize later).

## Prototype-first workflow

Before any C++ was written, the execution model (warps, lanes, masking,
shared memory, the IPDOM reconvergence stack) was built and validated as a
standalone Python prototype in `simulation/` (`simt_model.py`, `kernels.py`,
`trace.py` for cycle-by-cycle visualization). That let the semantics get
pinned down -- and bugs like an off-by-one boundary comparison in a
divergent-branch kernel get found -- without RISC-V encoding or a compiler
in the way. The C++ port (`32I/include/SIMT.h`/`src/SIMT.cpp`) mirrors that
validated model 1:1, just with real instruction words and `CPU`'s own
decode/ALU/branch logic instead of symbolic ops. See `simulation/README.md`.

## Phase 1 -- register file

`SIMTCore` owns a register file sized `T * W * 32` (4096 at T=32, W=4).
Storage is banked **by lane**, not by flat address:
```
regfile[thread_lane][warp_id][arch_reg_num]
```
so all T lanes can read their operands for the same warp in one (simulated)
cycle -- lane `t` only ever touches bank `t`. `x0` has no storage: reads
return 0, writes are dropped, for every thread independently.

`physical_addr(warp, thread, reg) = warp*T*32 + thread*32 + reg` is a flat
logical numbering for debug dumps/assertions only -- it is **not** how the
storage is laid out.

## Phase 2 -- ALU lanes

One decoded instruction is broadcast to every lane whose `tmask` bit is
set; each lane reads its own operands from its own bank and writes its own
result back. No new arithmetic opcodes: `SIMTCore` reuses `CPU`'s existing
RV32IM decode/ALU/branch logic directly, via a narrow `friend class
SIMTCore;` grant in `CPU.h` -- not inheritance (`SIMTCore` has no single
`pc`/`registers[32]` the way `CPU` does, so "is-a CPU" would be the wrong
relationship; see progress log for the fuller reasoning). Three static,
stateless `CPU` methods are reused:
- `CPU::decode_one(word)` -- same decoder the superscalar path already uses.
- `CPU::alu_exec(f, a, b, pc)` -- OP-IMM/OP (incl. M-extension)/LUI/AUIPC,
  extracted from `CPU::execute_one` into a pure function (`a`/`b` are the
  already-read rs1/rs2 values) so lane code and the scalar/superscalar
  paths share one implementation.
- `CPU::branch_taken(funct3, a, b)` -- BEQ/BNE/BLT/BGE/BLTU/BGEU.

Loads/stores go through the single shared `Memory*` (no per-SM scratchpad
yet -- flagged as later work), each lane computing its own address from its
own registers.

## Phase 3 -- SIMT control ISA (custom-1, opcode `0x2B`)

Same R-type shape as the NPU's custom-0: `.insn r 0x2B, funct3, 0, rd, rs1,
rs2`.

| funct3 | mnemonic | rd | rs1 | rs2 | semantics |
|---|---|---|---|---|---|
| 0 | `TMC`    | -       | count       | -                | `tmask = (1<<count)-1` for the current warp |
| 1 | `WSPAWN` | -       | count       | addr             | **not implemented, faults** -- needs Phase 4 (activates *other* warps) |
| 2 | `TID`    | dest    | -           | -                | `rd = pack_tid(warp, thread)` for the calling thread |
| 3 | `BAR`    | -       | id          | count            | **not implemented, faults** -- needs Phase 4 (multi-warp rendezvous) |
| 4 | `SPLIT`  | -       | predicate   | reg holding reconv_pc | push `(old_mask, reconv_pc)`; narrow `tmask` to lanes where predicate != 0 |
| 5 | `JOIN`   | -       | -           | -                | pop IPDOM stack; restore `tmask` |
| 6 | `PRED`   | -       | predicate   | -                | AND `tmask` with per-lane predicate, no stack push (caller restores manually) |
| 7 | `IDENT`  | dest    | -           | -                | `rd` = identity value chosen by funct7: 0 blockIdx, 1 blockDim, 2 gridDim, 3 hw_tid (other selectors fault) |

`TID` packing (`SIMTCore::pack_tid`, this prototype's own choice -- the
spec left the bit layout open): `[23:16] = warp*T+thread` (flat index),
`[15:8] = warp_id`, `[7:0] = thread_id`.

**SPLIT/JOIN is a predication model, not the full two-path hardware SIMT
reconvergence stack.** `SPLIT` pushes the *old* (pre-branch) mask and
narrows to the taken lanes; it never jumps to a second PC for the untaken
half. An `if/else` is therefore two separate `SPLIT`/body/`JOIN` blocks in
program order -- first `SPLIT(pred)`/if-body/`JOIN`, then
`SPLIT(NOT pred)`/else-body/`JOIN` -- not one `SPLIT` branching to two
different code paths. This matches the spec's literal wording ("push old
mask... apply predicate as new mask") and keeps the mechanism simple;
`reconv_pc` is carried in the stack entry but not functionally used by
`JOIN` today (no assert yet that `pc == reconv_pc` at `JOIN` time -- noted
as a possible future sanity check, see progress log).

`SPLIT`'s `rs2` (reconv_pc) and `TMC`'s `rs1` (count) are read from the
warp's **leader lane** (`SIMTCore::leader`, the lowest-numbered active
lane) rather than checked for agreement -- both are expected to hold a
compile-time-uniform constant loaded identically into every active lane
beforehand.

**`BRANCH` (plain RV32I `0x63`, not a SIMT-specific opcode) is
SIMT-uniform-only**: `issue()` requires every active lane to agree on the
branch outcome, and asserts otherwise with a message pointing at
`SPLIT`/`JOIN`. A condition that can differ per lane must go through
`SPLIT`/`JOIN` (or `PRED`), not `BRANCH`. `JAL` is fully uniform (no
register inputs to its target at all). `JALR`'s target is computed from
the leader lane's `rs1` only, with **no** cross-lane agreement check --
documented as a known simplification (no current kernel/toolchain-emitted
code produces a per-lane-divergent `JALR` target).

## Grid launch (host CPU -> GPU)

- **Host side:** `LAUNCH` is custom-2, opcode `0x5B` (`.insn r 0x5B, 0, 0,
  x0, rs1, rs2`): rs1 = kernel entry, rs2 = `(num_blocks << 16) |
  threads_per_block`, and the kernel argument is read implicitly from `a0`.
  The CPU stalls in that instruction until the whole grid has run, adds the
  device cycles to its own `cycle_count` (64-bit), and resumes at pc+4. Other
  funct3 values are reserved (error + ignored). With no GPU attached
  (`CPU::attach_gpu()`), LAUNCH prints an error and does nothing.
- **Superscalar CPU:** a LAUNCH always issues **alone** in its cycle
  (`hazard_scan`): everything before it has committed (incl. whatever set
  `a0`), nothing after it shares the cycle, and it is never speculative.
  Scalar and superscalar share `CPU::run_launch()`.
- **`GridLauncher`** owns `NUM_SMS` = 4 `SIMTCore`s sharing one `Memory*`.
  Block b runs on SM `b % NUM_SMS`; blocks on one SM run back to back. The
  launch's device cycles = the busiest SM's total (SMs are parallel in
  hardware; the emulator runs them one after another). `threads_per_block`
  must be 1..128 (one SM's warp slots), `num_blocks` 1..65535.
- **Stacks:** the CPU owns the host stack size (`CPU::HOST_STACK_RESERVE`,
  64KB at the top of RAM) and passes it to `GridLauncher`'s constructor;
  every lane's 1KB stack is carved below it by its unique hardware thread id.
  RAM too small for the reserve plus all device stacks (512KB) is rejected up
  front. The launcher also seeds `ra` (a return sentinel that ends the warp)
  and `a0`, so a kernel is a plain C function `void kernel(void* args)` with
  no startup stub.
- **Faults:** an SM lane may only run what `SIMTCore::unsupported_name()`
  allows (RV32IM, FENCE as a no-op, the SIMT ops except WSPAWN/BAR). Anything
  else -- LAUNCH, NPU `0x0B`, ECALL/CSR, WSPAWN, BAR, a bad IDENT selector,
  invalid load/store/branch/JALR widths, unknown opcodes -- prints
  `[SIMTCore Error] ... unsupported instruction`, raises the fault flag and
  stops the block; `GridLauncher::faulted()` abandons the rest of the grid;
  the host CPU halts after a LAUNCH that faulted or timed out (a block that
  hits the per-block issue cap).
- **Tracing:** `assemblyinstruction -trace` nests the GPU's work under the
  host's LAUNCH line (SM, block, warp, SM-local cycle, pc, lane mask) --
  `scripts/README.md`.

## Phase 4 -- scheduler (not started)

Planned: round-robin across the W resident warps, single-issue (one warp's
instruction per cycle, matching Vortex's design), a scoreboard tracking
in-flight/busy registers per warp, and a tunable `LOAD_LATENCY` (default 0
= every load completes the cycle it issues) that defers a destination
register's ready bit by N cycles via an inflight-tracker keyed by
`(warp_id, dest_reg)`. Needed before `WSPAWN`/`BAR` mean anything (both
fault today). Until then a block's warps run one after another to
completion (`SIMTCore::run_block`), which is only correct for kernels with
no barriers.

## Phase 5 -- NPU handoff (not started)

Deferred by design, per the original spec: how a SIMT thread triggers the
existing custom-0 NPU instruction (likely one thread of a warp issuing it,
whole warp stalled until done) is an open decision, not yet built. Today, a
custom-0 opcode (`0x0B`) issued from a SIMT lane **faults** (see "Grid
launch" above) rather than silently doing nothing.

## Toolchain

No compiler/assembler modifications. New instructions are emitted via
`.insn` inline asm wrapped in C functions (`tests/simt/simt_isa.h`), exactly
like `tests/npu/c_tests/matmul_template_2.c`'s `npu_load_a`/etc. already do
for the NPU, so kernels remain ordinary C compiled by the stock
`riscv64-unknown-elf-gcc` flow.

**Two things a SIMT kernel's C source must get right that a plain scalar
kernel never has to** (see progress log for how these were found):

1. **Every `simt_*` wrapper is `__attribute__((always_inline))`, not just
   `static inline`.** This is required for correctness, not performance.
   `static inline` alone is a hint GCC ignores at `-O0`, so these would
   otherwise compile to real `jal`/`ret` calls -- fatal specifically for
   `simt_split`/`simt_join`, since `SPLIT` narrows the active mask *while
   some lanes are still inside that call frame*. Masked-off lanes freeze
   with `sp`/`s0`/`ra` pointing at the callee's frame; when `JOIN` later
   reactivates them, the shared warp `pc` has moved on to wherever the
   surviving lanes got to (often a different call-nesting depth), so they
   read memory relative to a frame pointer that no longer matches what that
   `pc` expects. Forcing real inlining keeps the whole `SPLIT..JOIN` region
   as straight-line code in the *caller's* frame, so there's no nested
   frame for a masked-off lane to get stuck inside.
2. **Every lane needs its own private stack**, carved out of the single
   shared `Memory*` by thread id. For launched kernels the launcher sets
   every lane's `sp` itself (see "Grid launch"); the old single-warp
   harness path still uses `tests/simt/start_simt.s` (not the scalar
   `tests/python/start.s`) for this. At `-O0`, even the simplest C
   function spills locals to the stack via `sp`/`s0` (frame-pointer-relative
   stores) -- if every lane's `sp` pointed at the same address (as the
   scalar `start.s`'s single shared `_stack_top` does), those spills would
   silently clobber each other across lanes within the same broadcast
   cycle. `start_simt.s` reads the real `TID` instruction before anything
   else runs and offsets `sp` by `(thread_id+1) * 0x400` from `_stack_top`,
   giving each of the (up to 32) lanes its own 1KB slice.

An ordinary C `if` compiles to a real conditional branch, and `BRANCH` is
SIMT-uniform-only (see above) -- so a divergent condition must be wrapped in
explicit `simt_split()`/`simt_join()` calls, branching on the value
`simt_split()` **returns** (not on the original condition):
```c
if (simt_split(v < 0)) out[i] = v / 8;   // if
simt_join();
if (simt_split(v >= 0)) out[i] = v;      // else
simt_join();
```
`simt_split()` passes its predicate through the asm (`"+r"`), so the
compiler can't merge the two regions into one branch (gcc -O2 did exactly
that). See `tests/simt/grid_tests/c_relu.c`.

3. **SIMT programs are compiled at `-O0` only, for now** (`simt_isa.h` stops
   an optimised build with `#error`). The optimiser assumes the lanes that
   fail a `simt_split()` condition run their own path, and may move code
   there -- e.g. compute a default value only on that path. On this hardware
   those lanes are switched off until `simt_join()` and run nothing, so the
   value is never computed (22/64 lanes right at -O2 in review). Even at
   `-O0`, keep to the rule that a region passes results out **through
   memory**, not through a variable assigned inside it and read after the
   `simt_join()`. The real fix is a compiler that places SPLIT/JOIN itself.

**C API** (`tests/simt/simt_isa.h`): `simt_launch(kernel, threads_per_block,
num_blocks, args)` on the host (`kernel<<<blocks, threads>>>(args)`);
`simt_thread_idx()`, `simt_block_idx()`, `simt_block_dim()`,
`simt_grid_dim()`, `simt_global_id()`, `simt_hw_tid()` in kernels; plus
`simt_split`/`simt_join`/`simt_pred`/`simt_tmc`. Assembly programs use the
same instructions through `tests/simt/grid_tests/simt_macros.inc`.

Verified against the real `riscv64-unknown-elf-gcc` for four kernels
(`tests/simt/c_tests/*.c`, driven by `tests/simt/run_simt_tests.py` --
same methodology as `tests/npu/run_npu_tests.py`: compile, run through
`tests/simt/harness.cpp`, dump memory, diff against an
independently-computed expected result in Python): `scalar_multiply`,
`divergent_branch`, `tmc_masking`, `leaky_relu_32` (32 threads, 1 SM).

Grid launches are tested by `tests/simt/run_grid_tests.py` (`make -C 32I
run-grid-tests`, also part of `run-simt-tests`): hand-written assembly
kernels and host programs (`tests/simt/grid_tests/*.s`) and C programs with a
host `main()` and kernels (`grid_tests/c_*.c`), run through
`tests/simt/grid_harness.cpp` and `emul`'s own harness, diffed against
values computed independently in Python, with stderr checked too. Every
host test reruns on the superscalar CPU at issue widths 1, 2, 4 and 10.
