# Grid launch -- host CPU launches kernels onto a multi-SM SIMT device

**Status: implemented (phases A-F, branch `cuda_prototype`).** Where the
build differs from this design, the current rules are in `docs/CUDA/plan.md`
("Grid launch") and the reasons in `docs/CUDA/progress.md` section 6 -- in
short: the CPU owns the host stack reserve and passes it to `GridLauncher`;
an instruction an SM can't run faults (flag -> grid abandoned -> host CPU
halts) instead of being a silent no-op; tests are assembly + C programs
driven by `tests/simt/run_grid_tests.py`; C SIMT programs must be built at
`-O0`; and the divergence pattern is `if (simt_split(c)) {...} simt_join();`.

This extends
`docs/CUDA/plan.md` (the single-warp SIMT engine, Phases 1-3) with the
missing piece for practical use: a *program* -- ordinary C compiled for the
scalar `CPU` -- launches a kernel itself, CUDA-style, instead of a C++ test
harness hand-driving `SIMTCore::reset_warp()`/`issue()`.

## 1. Goal and non-goals

**Goal.** A user writes one C file containing a host `main()` and one or more
kernel functions. `main()` calls `simt_launch(kernel, threads_per_block,
num_blocks, args)`. The scalar CPU stalls, the "GPU" runs the whole grid,
control returns to the CPU at the next instruction, and the host reads the
results straight out of the shared `Memory`. Same unmodified
`riscv64-unknown-elf-gcc`, new instructions via `.insn` -- exactly like the
NPU's custom-0.

**Non-goals for this increment** (all deliberately deferred):
- Warp scheduler / latency modelling (Phase 4). There is no latency in the
  model, so warps simply run to completion one after another.
- `BAR` / `WSPAWN` (need real warp interleaving). Kernels launched this way
  must not use them.
- Non-blocking launch, multiple kernels in flight, streams.
- A per-SM shared scratchpad (still the flat shared `Memory*`).
- NPU handoff from inside a kernel (Phase 5).

## 2. Mental model

Three layers, mirroring a real GPU:

```
 host CPU  (scalar/superscalar `CPU`, runs main())
    |   LAUNCH instruction (opcode 0x5B)  -- CPU stalls here
    v
 GridLauncher   -- the block distributor. Owns NUM_SMS SMs. Assigns block b -> SM (b % NUM_SMS).
    |
    v
 SM 0 .. SM N-1 (= `SIMTCore`, one object each)  -- splits a block into warps, runs them
    |
    v
 warps -> 32 lanes each (existing lockstep/tmask engine, unchanged)
```

- **One launch = one kernel = one grid.** All blocks start at the same entry
  address; every thread is unique, identified by `(blockIdx, threadIdx)`.
- **Blocks are independent.** No ordering, no cross-block sync, communicate
  only through memory. That independence is what lets the emulator run them
  sequentially while still modelling "different SMs".
- **A block never spans SMs.** An SM splits it into `ceil(threads_per_block/32)`
  warps; the last warp is partially masked if the count isn't a multiple of 32.
- **Launch is blocking.** The CPU does not fetch another instruction until the
  grid has finished (same as the NPU's instructions today).

Why no warp scheduler is needed yet: the only reasons to interleave warps are
hiding latency and barriers. The model has neither, so "run warp 0 to
completion, then warp 1, ..." is functionally identical to any interleaving
(for barrier-free kernels, which is all this increment supports).

## 3. Configuration (compile-time constants, same posture as `SIMT.h`)

| constant | value | where | note |
|---|---|---|---|
| `THREADS_PER_WARP` | 32 | `SIMT.h` | unchanged |
| `WARPS_RESIDENT` | 4 | `SIMT.h` | unchanged; caps `threads_per_block` at 128 |
| `NUM_SMS` | 4 | new, `GridLauncher.h` | number of `SIMTCore` instances |
| `MAX_THREADS_PER_BLOCK` | `THREADS_PER_WARP * WARPS_RESIDENT` = 128 | derived | a block must fit one SM's warp slots |
| `MAX_BLOCKS` | 65535 | derived from the 16-bit field | |
| `STACK_BYTES_PER_THREAD` | 0x400 | new | same 1KB/lane as `start_simt.s` today |
| `HOST_STACK_RESERVE` | 0x10000 | new | top 64KB of RAM left to the host's stack |

## 4. ISA

### 4.1 Host side: `LAUNCH` -- new custom-2 opcode `0x5B`

Deliberately a *different* opcode from the device-side custom-1 (`0x2B`)
ops, so "executed by the CPU" and "executed by an SM lane" never share an
encoding. R-type: `.insn r 0x5B, funct3, 0, x0, rs1, rs2`.

| funct3 | mnemonic | rs1 | rs2 | implicit | semantics |
|---|---|---|---|---|---|
| 0 | `LAUNCH` | kernel entry address (function pointer) | `(num_blocks << 16) \| threads_per_block` | reads host `x10`/`a0` as the kernel argument | run the grid to completion, then resume at pc+4 |

- `a0` is read implicitly (the way `ecall` implicitly reads `a7`), because
  R-type only has two source registers and the kernel needs three inputs. The
  `simt_launch()` macro pins the argument to `a0` with a register-asm
  constraint. (Rejected alternative: R4 format `.insn r4` for a real third
  operand -- needs `decode_one()` to expose `rs3`, more surface for one
  value.)
- Other funct3 values: reserved. Executing one is a no-op plus a stderr
  message, like `Memory`'s existing error style.
- Launch validation (any failure: print `[GridLauncher Error] ...` to stderr,
  skip the launch, host continues): `threads_per_block == 0`,
  `threads_per_block > MAX_THREADS_PER_BLOCK`, `num_blocks == 0`.

### 4.2 Device side: changes to custom-1 (`0x2B`)

`TID` (funct3 2) changes meaning slightly; one new funct3 (7) is added.

**`TID` new packing** (the old one only made sense for a lone warp 0):

| bits | field |
|---|---|
| `[7:0]` | lane id within the warp (0..31) -- **unchanged**, so today's `simt_tid() & 0xFF` kernels keep working for single-warp runs |
| `[15:8]` | warp index *within the block* |
| `[23:16]` | `threadIdx` = flat thread index within the block = `warp_in_block*32 + lane` |

**Identity ops: funct3 7, `funct7` selects the value**, `rd` receives it
(`.insn r 0x2B, 7, <funct7>, rd, x0, x0`), identical for every active lane
except `HW_TID`:

| funct7 | name | value |
|---|---|---|
| 0 | `BLOCK_IDX` | `blockIdx` |
| 1 | `BLOCK_DIM` | `threads_per_block` |
| 2 | `GRID_DIM` | `num_blocks` |
| 3 | `HW_TID` | unique physical thread id = `(sm*WARPS_RESIDENT + warp_slot)*THREADS_PER_WARP + lane` -- only needed by the runtime (stack carving, debugging) |

`funct7` is already exposed by `CPU::decode_one` (`InstructionFields.funct7`),
so no decoder change is needed. Values fit well within its `int8_t`.

### 4.3 C-side API (`tests/simt/simt_isa.h`, all `always_inline`)

```c
// device (inside a kernel)
uint32_t simt_thread_idx(void);   // threadIdx within the block
uint32_t simt_block_idx(void);
uint32_t simt_block_dim(void);
uint32_t simt_grid_dim(void);
uint32_t simt_global_id(void);    // block_idx * block_dim + thread_idx

// host (inside main)
void simt_launch(void (*kernel)(void*), uint32_t threads_per_block,
                 uint32_t num_blocks, void* args);
```

A kernel is a normal C function `void kernel(void* args)`. The
`always_inline` rule for every `simt_*` wrapper from `plan.md` still applies
(SPLIT/JOIN correctness) -- `simt_launch` is host-only and is the one
exception where a real call would be harmless, but keep it inline for
consistency and so `a0` pinning is local.

## 5. Components and changes

### 5.1 `SIMTCore` (the SM) -- `include/SIMT.h`, `src/SIMT.cpp`

Existing: banked regfile, `Warp` state, `issue()`, `exec_lane()`,
`exec_simt()`, `reset_warp()`. New:

- **`sm_id`** member, set by the constructor (`SIMTCore(Memory*, int sm_id = 0)`
  keeps existing tests compiling).
- **Per-warp context** added to `Warp`: `block_id`, `warp_in_block`; per-SM
  block context: `block_dim`, `grid_dim`.
- **`BLOCK_RETURN_PC` sentinel** (e.g. `0xFFFFFFF0`, outside RAM): after
  `issue()` sets `warp.pc = next_pc`, `if (warp.pc == BLOCK_RETURN_PC)
  warp.halted = true`. This is how a kernel function's `ret` ends a warp. The
  existing self-jump halt idiom stays, so the four current tests are
  unaffected.
- **`exec_simt()`**: `TID` repacked as in 4.2; new `funct3 7` handler reading
  `f.funct7`. Needs a per-lane write for `HW_TID`, uniform write for the rest.
- **`uint64_t run_block(entry, block_id, block_dim, grid_dim, args)`** -- the
  one new public entry point, returns cycles spent (instructions issued):
  1. `nwarps = ceil(block_dim / 32)`; for each warp `w < nwarps`:
     - `reset_warp(w, entry)`; if `w` is the last warp and `block_dim % 32 != 0`,
       narrow `tmask` to `(1 << (block_dim % 32)) - 1`.
     - Zero the warp's registers for all 32 lanes (deterministic start, no
       stale state from the previous block on this SM).
     - Per active lane initialise `sp` = this thread's private stack top
       (5.3), `ra` = `BLOCK_RETURN_PC`, `a0` = `args`.
     - Record `block_id`, `warp_in_block = w`.
  2. For `w = 0..nwarps-1`: `while (!warp_halted(w)) issue(w)` (run to
     completion; `max_cycles` safety cap -> error message, abandon block, so a
     runaway kernel can't hang the host forever).
  3. Return the total instructions issued.
- `reset_warp()` and `issue()` stay as they are; the existing harness-style
  usage keeps working.

### 5.2 `GridLauncher` (new) -- `include/GridLauncher.h`, `src/GridLauncher.cpp`

```
class GridLauncher {
  GridLauncher(Memory*);                    // constructs NUM_SMS SIMTCores sharing the Memory*
  uint64_t launch(entry, threads_per_block, num_blocks, args);   // returns device cycles
  void set_verbose(bool);                   // print block->SM map + per-SM cycles
};
```

`launch()`:
1. Validate (4.1).
2. `sm_cycles[NUM_SMS] = {0}`.
3. `for b in 0 .. num_blocks-1`: `sm = b % NUM_SMS`;
   `sm_cycles[sm] += sms[sm].run_block(entry, b, threads_per_block, num_blocks, args)`.
   (Sequential in the emulator; "parallel" only in the accounting below.)
4. Return `max(sm_cycles)` -- SMs run in parallel in hardware, blocks on one
   SM run back to back. (`+ LAUNCH_OVERHEAD`, constant 0 for now.) This is a
   *modelling choice*, not a correctness property; revisit with Phase 4.

Block->SM policy is a deliberately trivial round-robin so it is easy to swap
(e.g. least-loaded) later; that swap touches only step 3.

### 5.3 Stacks (replaces `start_simt.s` for launched kernels)

Per-lane private stacks are still mandatory (plan.md "two things a SIMT
kernel must get right"). Instead of a startup stub reading `TID`, **the
launcher initialises `sp` itself** -- like the CUDA runtime allocating local
memory per thread -- which means a launched kernel is just a plain C function
with no asm entry stub at all.

```
device_stack_top   = memory->size() - HOST_STACK_RESERVE
thread's sp        = device_stack_top - (hw_tid + 1) * STACK_BYTES_PER_THREAD
hw_tid             = (sm_id * WARPS_RESIDENT + warp_slot) * THREADS_PER_WARP + lane
```

- Unique across SMs, warps and lanes, so concurrent-semantics-safe even though
  the emulator runs blocks sequentially.
- Total = `NUM_SMS * WARPS_RESIDENT * 32 * 1KB` = 512KB below the host's 64KB
  reserve. With the 4MB RAM: device stacks ~3.43MB..3.94MB, host stack
  ~3.94MB..4MB, program/data below. Needs a **new `Memory::size()`
  accessor** (tiny public addition to `memory.h`).
- Still unguarded: no overflow check between adjacent slices (existing open
  item in `progress.md`).
- Alternative considered and rejected: keep `start_simt.s` as a launch stub.
  It would need the stub's address passed to the launcher, an extra
  indirection (stub calls the kernel pointer), and a halt-loop convention.
  Launcher-initialised state is less code and simpler for users.

### 5.4 Host `CPU` -- `include/CPU.h`, `src/CPU.cpp`

- `class GridLauncher;` forward declaration in `CPU.h` (avoids the include
  cycle: `SIMT.h` includes `CPU.h`); `CPU.cpp` includes `GridLauncher.h`.
- `GridLauncher* gpu` member (null by default) + `void attach_gpu(GridLauncher*)`.
  Executing `0x5B` with no GPU attached prints an error and no-ops.
- **Scalar path:**
  - `decode()`: opcode `0x5B` -> `mem_read/write/reg_write_enable = false`
    (same shape as the `0x0B` case).
  - `execute()`: `aluResult = registers[rs1]` (kernel entry), like `0x0B`.
  - `read()`: where the `0x0B` block lives -- decode `rs2` into
    `(num_blocks, threads_per_block)`, call `gpu->launch(entry, tpb, nb,
    registers[10])`, `cycle_count += returned cycles`. Nothing is written back.
  - The CPU naturally "stalls" -- `read()` simply doesn't return until the
    launch is done; no new state machine.
- **Superscalar path** (second step, after the scalar path is green):
  - `decode_all()`: same control signals; `execute_one()`: same `aluResult`;
    `read_one()`: same launch call.
  - `hazard_scan()`: a launch **issues alone** -- `if (i > 0) issueCount = i;
    else issueCount = 1; return`. Reasons: it has a hidden `a0` source (RAW),
    it has irreversible side effects (must never be a speculative slot, and
    slot 0 never is), and it is a full memory barrier. Add `0x5B` to
    `reg_usage()` as using rs1+rs2.
- No change to the halt idiom, `next_pc`, or `writeback()`.

### 5.5 `Memory` -- `include/memory.h`, `src/memory.cpp`

Only `size_t size() const { return mem_array.size(); }`. The GPU is *not* part
of `Memory` (unlike the NPU): the NPU is a passive device behind an address
window, whereas `SIMTCore` is an active bus master that fetches and
loads/stores through `Memory`; owning it there would make `Memory` and
`SIMTCore` own/point at each other.

## 6. Memory map used by the tests (4MB RAM, `link.ld` unchanged)

```
0x000000  host + kernel code (.text), .data, .bss          (single binary, _start first)
0x100000  result region (RESULT_BASE, same as today's SIMT tests)
   ...    (free)
~0x370000 device stacks: 512KB, carved top-down from (RAM_end - 64KB)
~0x3F0000 host stack (top 64KB), host sp starts at _stack_top
```

Host entry is the existing scalar `tests/python/start.s` (`la sp,_stack_top;
call main`). Kernel functions are linked into the same `.text` and are only
ever *executed* by SM lanes.

## 7. Implementation phases (each ends in a green gate)

| # | work | gate |
|---|---|---|
| A | `SIMTCore`: `sm_id`, `Warp`/block context, `BLOCK_RETURN_PC`, new `TID` packing, `funct3 7` identity ops, `run_block()` | all 4 existing `run_simt_tests.py` kernels and `simt_regfile_test` still pass unchanged; new C++ unit test calls `run_block()` directly on a hand-loaded kernel |
| B | `GridLauncher` + `Memory::size()` | C++ test: launch over `NUM_SMS`-wrapping block counts, check block->SM map and per-SM/max cycle numbers |
| C | host `CPU` scalar path: opcode `0x5B`, `attach_gpu()`, `cycle_count` | end-to-end C test (below) passes; `tests/python/run_tests.py` and `run_npu_tests.py` unaffected |
| D | `simt_isa.h` macros (`simt_launch`, identity helpers), new `tests/simt/grid_harness.cpp` + `run_grid_tests.py` + Makefile targets (`run-grid-tests`) | full grid test matrix (section 8) passes |
| E | superscalar path (`hazard_scan` rule etc.) | the grid tests also pass with `mode=superscalar` at several issue widths, same expected output |
| F | docs: update `plan.md` (replace the Phase 4/5 stubs' relevant text), add a progress-log entry, README note; only touch `CLAUDE.md` if asked | -- |

Phase A deliberately changes nothing observable for the existing tests: the
old harness still calls `reset_warp(0, 0)` + `issue(0)`, `start_simt.s` still
reads `TID`'s low byte, and `reset_warp()` is untouched.

## 8. Test plan

New harness `tests/simt/grid_harness.cpp` (builds `build/grid_harness`): create
`Memory(4MB)`, a `GridLauncher`, a `CPU` with `attach_gpu()`, load the
host+kernel binary, run the CPU loop to halt or cycle cap, dump the result
region to a file. `tests/simt/run_grid_tests.py` follows
`run_simt_tests.py` exactly: cross-compile `start.s` + the C file, run, parse
the dump, diff against an **independently computed** expected list in Python.
Kernels live in `tests/simt/grid_tests/`.

| test | what it proves |
|---|---|
| `grid_vecadd` | 4 blocks x 64 threads (256 elements): `out[gid]=a[gid]+b[gid]`; basic launch, args struct via `a0`, `global_id` math, 2 warps/block |
| `grid_partial_warp` | 3 blocks x 40 threads: last warp has 8 live lanes; sentinel-seeded output proves masked lanes (40..63 of each block) never write |
| `grid_more_blocks_than_sms` | 9 blocks x 32 on `NUM_SMS`=4: kernel writes `blockIdx` and the `HW_TID`-derived SM id; Python checks `sm == block % NUM_SMS` and that all 9 blocks ran exactly once |
| `grid_identity_ops` | every thread writes `(blockIdx, threadIdx, blockDim, gridDim)`; exact tuple per global thread |
| `grid_two_launches` | launch A then launch B where B consumes A's output; host code *between* and *after* launches (sums the results into memory) -- proves the CPU resumes at the right pc and sees device writes |
| `grid_divergent_relu` | `leaky_relu` across blocks with `SPLIT/JOIN` -- divergence still correct under the block path |
| `grid_bad_launch` | `threads_per_block = 129` and `num_blocks = 0`: launch skipped, host continues, a "host-ran-after" marker is written, no device output |
| existing 4 SIMT kernels | regression: unchanged results |

Extra assertions the Python driver makes beyond result values: no unexpected
`[GridLauncher Error]`/`[Memory Error]` on stderr in the passing tests, and
the launch's reported device cycles equal `max` over SMs of the per-SM
instruction counts (via `set_verbose`).

## 9. Risks and open questions

1. **Run-to-completion assumes no barriers.** A kernel using `BAR` would
   deadlock or compute garbage once `BAR` exists. Mitigation now: `BAR`
   stays a no-op *and* documented as unsupported inside launched kernels.
   Real fix = Phase 4's "switch warp when blocked" rule.
2. **`simt_tmc()` inside a launched kernel** re-widens/narrows the mask in a
   way the launcher's partial-warp mask doesn't know about. Rule: launched
   kernels must not use `TMC`; use `blockDim`/`threadIdx` bounds via
   `SPLIT`/predicates instead. (`scalar_multiply`/`tmc_masking` stay as
   single-warp harness tests.)
3. **Unguarded stack slices** (1KB/lane): a deep kernel silently corrupts its
   neighbour's stack. Could add a cheap check in `exec_lane` that `sp` stays
   inside the thread's slice.
4. **Cycle model** (sum within an SM, max across SMs) is a placeholder;
   numbers are not comparable to real hardware until Phase 4.
5. **Divergent `JALR`** target (leader-lane only) now also matters for kernel
   `ret`: it works because every lane's `ra` is identical
   (`BLOCK_RETURN_PC`), but a kernel that does indirect calls through
   per-lane pointers is still unchecked.
6. **Decisions made here that you may want to change:**
   - launcher-initialised `sp/ra/a0` instead of a `start_simt.s`-style stub (5.3);
   - host launch opcode `0x5B` (custom-2) rather than reusing `0x2B` (4.1);
   - kernel argument passed implicitly in `a0` (4.1);
   - `NUM_SMS = 4` and round-robin `block % NUM_SMS`;
   - invalid launches are a stderr message + skip, not a trap (this codebase
     has no trap mechanism).
