# RSVP — RISC-V CPU Emulator

A C++ behavioral emulator for RISC-V, built incrementally extension by extension.
Goal right now: get a working *functional* model executing instructions correctly.
Timing/microarchitecture accuracy is not a current goal — "pipeline" below refers to a
sequential stage-by-stage instruction lifecycle, not an overlapped/hazard-aware HW pipeline.

## Layout

- `32I/` — complete RV32I base integer ISA + M-extension (mul/div), plus an in-progress NPU
  (matrix-multiply accelerator) — see "NPU" below. Reference implementation.
- `32IV/` — RV32I + V-extension (vector) work in progress, built on top of a copy of 32I.
- `ramulator` — external memory simulator, vendored/unrelated to current focus.
- `scripts/` — `emul` and `assemblyinstruction`, standalone CLI helpers (symlinked into
  `~/.local/bin`) for running or disassembling a single C file against 32I/32IV without
  hand-rolling the cross-compile/link/run steps each time. See `scripts/README.md`.

Each extension directory is self-contained and independently buildable:
```
<ext>/
  include/CPU.h, memory.h[, CPU_Vector.h][, NPU.h (32I only)]
  src/CPU.cpp, memory.cpp, main.cpp[, CPU_Vector.cpp][, NPU.cpp, NPU_print.cpp (32I only)]
  cmod/sc_cpu.cpp      # SystemC stub, not yet developed
  tests/basic/         # C test programs cross-compiled for rv32im and run on the emulator
  tests/python/        # run_tests.py: compiles each C test natively (ground truth) vs
                        # cross-compiled rv32im (run on the emulator), diffs the results
  tests/npu/           # 32I only — NPU matmul test generator, see "NPU" below
  Makefile             # `make` builds src/main.cpp demo; `make run-test` builds+runs
                        # tests/basic against CPU.cpp/memory.cpp[/NPU.cpp] directly
```
32IV currently duplicates 32I's CPU.cpp/memory.cpp and adds vector-specific pieces on top
rather than sharing code — expect near-identical diffs between the two trees for the
scalar (32I) parts.

## Core model (`CPU` class)

State: 32 general-purpose registers (`x0`-`x31`, `x0` hardwired to 0 via
`enforce_zero_register()`), `pc`/`next_pc`, a `cycle_count`, a `halted` flag (set by
`execute()` when a jump/branch targets its own address — this codebase's halt idiom, e.g.
`start.s`'s `_end: j _end`; queryable via `is_halted()`, overridable via `set_pc()` for
harnesses loading at a non-zero base), and a `Memory*` (flat byte-addressable
`std::vector<uint8_t>`, little-endian, bounds-checked read/write at byte/halfword/word
granularity).

Per-cycle stage sequence, called explicitly by the driving `main()` loop:
1. **fetch()** — read 32-bit instruction at `pc` from memory, default `next_pc = pc + 4`,
   reset control signals (`mem_read_enable`, `mem_write_enable`, `reg_write_enable`).
2. **decode()** — split the instruction into opcode/rd/rs1/rs2/funct3/funct7, compute all
   immediate encodings (U/I/S/B/J/Z types), and set control signals per opcode (this is a
   small control unit switch, not a hazard-aware decoder).
3. **execute()** — opcode-dispatched ALU/branch/jump logic. Nested `switch` on
   opcode → funct7 → funct3 (mirrors the RISC-V encoding structure). Branches/jumps update
   `next_pc` directly here; everything else produces `aluResult`.
4. **read()** — actual memory access stage (despite being called after execute): loads
   read from `memory` using `aluResult` as address into `memResult`; stores write
   `registers[rs2]` to `memory` at `aluResult`.
5. **writeback()** — commits `aluResult` or `memResult` (load path) into `registers[rd]`
   when `reg_write_enable`, re-enforces `x0 == 0`, and commits `pc = next_pc`.

Implemented so far: full RV32I base (arithmetic/logic/shift immediates and register-register
ops, LUI/AUIPC, loads/stores of all widths, all branches, JAL/JALR) plus the M-extension
(MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU) under opcode `0x33`/`funct7 0x01`.

## NPU (32I, in progress)

A memory-mapped matrix-multiply accelerator (`NPU` class — `include/NPU.h`, `src/NPU.cpp`,
`src/NPU_print.cpp`), owned as a member of `Memory` and addressed at `0x80000000`+. Computes
A (M×K) × B (K×N) → C (M×N), M/K/N each configurable up to the native tile size
`MAX_DIM`=16; A/B/C share one flat data array. Registers: `DIM_M/K/N_ADDR`, `TRIGGER_ADDR`
(one-shot `compute()`, overwrites C), `MAC_ADDR` (`mac()`, accumulates into C), `RESET_ADDR`
(zeroes dims/data/done), `STATUS_ADDR` (done flag), `PRINT_ADDR` (a *read* triggers
`print_matrices()`, a debug dump of A/B/C to stdout).

Wired into the CPU as a new custom-0 opcode (`0x0B`, `.insn r 0x0B, funct3, 0, x0, rs1,
rs2`), decoded like any other opcode but doing its actual work in `read()`, touching neither
a register nor the scalar mem_read/write path:
- funct3 0/1 (load A/B): copy a 16×16 tile as **one contiguous 256-word block** from `rs1`
  into the NPU's data window; `rs2` is unused. This requires the tile to already be
  contiguous in memory — see the tile-major layout below.
- funct3 2 (store C): a *strided* write — `rs2` = row stride in bytes — since the
  destination is a plain row-major matrix, not a contiguous tile.
- funct3 3: general-purpose debug instruction, unrelated to the NPU itself — prints an
  N×N (`rs2`=N) region of memory at `rs1` via a new `Memory::print_matrix()`.

`Memory` also gained `read_block` and two `dump_range` overloads (stdout, and a file-path
variant for test harnesses that need to read a region back programmatically).

Bigger-than-one-tile matmuls are tiled in software. `tests/npu/c_tests/` has two templates,
driven by `tests/npu/run_npu_tests.py` (random A/B, cross-compile, run, diff against an
independent Python matmul; `--template 1|2`, `--sizes`, `--seed`, `--low/--high`,
`--cycles`):
- `matmul_template.c` (`--template 1`) — original per-word MMIO `npu_write32` loop per tile.
- `matmul_template_2.c` (`--template 2`, default) — uses the custom-0 instructions above.
  Its A/B are generated **tile-major** (each 16×16 tile stored contiguously, tiles visited
  in row-major grid order — `run_npu_tests.py`'s `format_matrix_tiled_c`) specifically so
  the funct3 0/1 loads can be one contiguous access instead of 16 strided row reads; the
  result matrix is still written out plain row-major (funct3 2's strided store) since
  nothing downstream needs it tiled.

Also: `tests/basic/main.cpp` and `tests/python/link.ld` had RAM bumped 1M→4M (room for
larger tiled results at a fixed offset), and `main.cpp` gained optional argv dump-region
args (base/offset/file path) instead of unconditionally dumping a fixed region on every run.

A few standalone manual tests (`npu_matmul_test.c`, `npu_matmul_256by256.c`,
`npu_matmul_test_256.c`, run directly via `emul` rather than `run_npu_tests.py`) still have
header comments describing the *old* strided-load behavior for funct3 0/1 — stale, needs
updating to match the contiguous-load behavior above.

## Vector extension (32IV, in progress)

`CPU_Vector.cpp` adds vector-specific stages, invoked from `decode()`/`execute()`/`read()`/
`writeback()` when the opcode is a vector opcode (`0x57` OP-V, load/store `0x07`/`0x27`) via
two CPU-level control signals (`vector_mem_enable`, `vector_instruction_active`) that keep
the scalar mem/reg-enable paths from touching vector state, since `rd`/`vd` share bit
positions:
- **vector_decode()** — extracts common vector fields (vd/vs1/vs2, vm, vset immediates,
  bit30/31 for vsetvli/vsetivli/vsetvl), plus load/store-only fields (`nf`, `mew`, `mop`,
  EEW from `width`) and an `ls_supported`/`ls_whole_register`/`ls_fault_only_first` gate
  that routes anything not yet implemented into a safe no-op.
- **vector_config_execute()** — implements `vsetvli`/`vsetivli`/`vsetvl` (unchanged): decodes
  `vtype` (SEW, LMUL incl. fractional 1/8-1/2), computes `vlmax`, clamps AVL, sets `vl`/`vtype`.
- **vector_load_store_execute()** — per-lane (per-segment) address generation + active-lane
  mask (`vl` range and `v0.t` masking, applied at whole-segment granularity), covering
  unit-stride, strided (signed byte stride from a GPR), and indexed (offsets read from a
  vector register at the instruction's own EEW, data width = current SEW) addressing.
- **vector_read()** — the actual memory access: per-lane/per-field `Memory` reads or writes
  (byte-enabled, so masked-out bytes are never touched), a separate flat-block path for
  whole-register load/store (`vl<n>r.v`/`vs<n>r.v`, ignores `vl`/`vtype`/mask), and
  fault-only-first trimming (`vle<eew>ff.v`) approximated via `Memory::probe()` bounds
  checks — this codebase has no trap/exception mechanism anywhere, so an element-0 fault
  abandons the load instead of signaling a real trap.
- **vector_writeback()** — single commit point for all three vector opcodes: scalar `rd`
  write for config, vector register file commit for loads (incl. multi-field segment/
  whole-register commits), no-op for stores (already done in `vector_read()`).

`CPU_Vector.h`'s `VectorRegister` (32 × `VLEN_BYTES`=64-byte vregs) is now a real `CPU`
member (`vregfile`); `v0` doubles as the mask register. `Memory` gained `read_bytes`/
`write_bytes` (byte-enabled)/`probe` (silent bounds check) to support wide/masked vector
access. Not yet implemented: vector arithmetic ops (add/sub/etc.), mask load/store
(`vlm.v`/`vsm.v`), `LMUL>1` register grouping, `mew=1` (>64-bit elements).

Next planned work (per user): vector arithmetic instructions, then test cases.

## Testing

- `tests/basic/`: hand-loadable or cross-compiled `.bin` test programs run directly against
  `CPU`+`Memory` via `tests/basic/main.cpp`/`loader.cpp` (`load_binary` reads a raw binary
  into memory at a base address).
- `tests/python/run_tests.py`: the real correctness harness. For each C test file (listed in
  `run_tests_list.txt`, one filename per line) it (1) compiles natively as ground truth, (2)
  cross-compiles bare-metal for rv32im with `riscv64-unknown-elf-gcc`/`objcopy` using
  `start.s`/`link.ld`, (3) runs the resulting binary on the emulator (`build/run_test`), (4)
  compares the native result to register `x10` (a0) in the emulator's final state.
  `run_opt_tests.py` does the same sweep across `-O0`-`-O3` (list in `run_opt_tests_list.txt`).
- `32I/tests/riscv-arch-test/`: runs official riscv-arch-test instruction-level test files
  (`assembly_files/`, mirroring that repo's own `tests/<suite>/` layout) against this
  emulator. Bypasses the real framework's Sail/UDB/mise toolchain (too heavy for this repo's
  needs) via a hand-written `env/riscv_arch_test.h` implementing just the macros these tests
  use, and uses `qemu-riscv32` as the golden reference instead of Sail, diffing signature
  memory dumps (`run_arch_test.py`). See `README.md` there for the full manual-run walkthrough.
- `32I/tests/old/` exists but is superseded/untracked — don't treat it as current.
- `32I/tests/npu/run_npu_tests.py`: NPU matmul correctness harness — see "NPU" above.

---

## Updating this file

Only do the following when the user explicitly asks you to update CLAUDE.md (e.g. "update
CLAUDE.md", "refresh the context file") — never on your own initiative:

1. Re-read the current source files in `32I/` and `32IV/` (CPU.h/.cpp, CPU_Vector.h/.cpp,
   memory.h/.cpp, Makefiles, tests/) to see what has actually changed since this file was
   last written — diff against what's described here rather than assuming.
2. Review this conversation's history for any new architectural decisions, newly implemented
   instructions/extensions, changed pipeline behavior, or new test infrastructure.
3. Update the relevant sections above to reflect the current state — keep it at this level of
   detail (general structure, stage responsibilities, what's implemented vs. not) and do NOT
   expand into line-by-line or per-instruction explanations.
4. Keep this "Updating this file" section intact at the end.
