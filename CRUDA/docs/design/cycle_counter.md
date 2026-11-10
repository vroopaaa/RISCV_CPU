# Cycle counter: `rdcycle` / `rdcycleh` (32I hardware change)

The host CPU always counted its cycles (`CPU::cycle_count`), but a program could not
read the count. The harness printed one total when the program halted. That was enough
while each test ran a single lego. A transformer block runs about 20 lego calls in one
program, and accelerator design needs the breakdown: how many cycles go to the NPU
matmuls, how many to the GPU passes (softmax, layernorm, LUTs, adds), and which call to
make faster.

This change adds the two standard RISC-V counter reads, so a program can time each call
itself:

```c
uint64_t t0 = nn_cycles();
nn_layernorm(...);
uint64_t t1 = nn_cycles();
cycles[0] = (uint32_t)(t1 - t0);     // dumped by the driver with the outputs
```

Shared rules are in [00-overview.md](00-overview.md) §8.

## 1. Why the standard instruction

There were two options:

- **A memory-mapped counter register**, like the NPU's MMIO registers. Simple, but not
  standard: the same C code would not run on qemu or a real RISC-V core.
- **`rdcycle` / `rdcycleh`** (chosen). These are the RISC-V Zicntr counter reads. The
  assembler accepts them with `-march=rv32im`, every RISC-V core has them, and any
  existing timing code uses them.

Today, on this emulator, a lego's cost does not depend on what ran before it: there are
no caches, no memory model, and the GPU legos' loops are fixed-length and branch-free.
So the per-call numbers could also be had by running each call alone. The counter
becomes necessary once the memory hierarchy is modelled (a GEMM's cost will depend on
what is already in SRAM). It also lets one run produce the whole table.

## 2. The instructions

| Instruction | Encoding | Result |
|---|---|---|
| `rdcycle rd` | `csrrs rd, cycle, x0`, CSR `0xC00` (word `0xC0002073 \| rd << 7`) | low 32 bits of the cycle count |
| `rdcycleh rd` | `csrrs rd, cycleh, x0`, CSR `0xC80` | high 32 bits |

Both are SYSTEM-opcode (`0x73`) instructions with `funct3 = 2` and `rs1 = x0`.

**The value** is the number of cycles completed **before the cycle the read issues
in**:

- **Scalar host:** one cycle per instruction, so a read at instruction `i` returns `i`.
  The first instruction of a program reads 0. Two reads back to back differ by 1.
- **Superscalar host:** one cycle per issue group. Every read in a group returns the
  same value, the count before that group. Two reads in one group differ by 0. A read
  never waits for or blocks anything: it reads no register and has no memory class.
- **LAUNCH:** the CPU adds the grid's device cycles to its own count (the busiest
  SM's issued warp instructions, see `GridLauncher`). So a read after a LAUNCH includes
  them, and timing a GPU lego gives launch + kernel. A LAUNCH always issues alone on the
  superscalar host, so the device cycles are never added in the middle of a group.

The count is 64-bit (`cycle_count` already was). On RV32 reading it takes two
instructions, so a program that needs all 64 bits reads high, low, high again, and
retries if the high word changed (§4). 2³² cycles is about 4.3 billion. One ViT block,
or all 12, stays well below that, but the 64-bit read costs only a few instructions.

**Everything else on the SYSTEM opcode is unsupported**, as before: ECALL, EBREAK, CSR
writes (`csrrw`, and `csrrs` / `csrrc` with `rs1 ≠ x0`: the counters are read-only),
the immediate forms, and every other CSR (`instret`, `time`, the machine-mode CSRs).
Before this change the host silently ignored them. Now each one prints an error
(`[CPU Error] unsupported SYSTEM instruction 0x… at pc 0x… -- ignored`) and writes no
register, so a program that relies on one fails visibly instead of reading garbage.

**SM lanes** still fault on every SYSTEM instruction, `rdcycle` included. The counter
is the host CPU's. A per-SM `clock()` like CUDA's would be a separate change.

## 3. Changes in 32I

| File | Change |
|---|---|
| `include/CPU.h` | `OPCODE_SYSTEM`; `is_counter_read()`, `run_system()`; `set_cycle_count()` for harnesses |
| `src/CPU.cpp` | `decode()` / `decode_all()`: SYSTEM writes `rd` only for the two reads. `read()` / `read_m()`: `run_system()` returns the count's half or prints the error. `cycle_count` advances only in `writeback()` / `writeback_m()`, so the value read is the count before this cycle |
| `tests/basic/main.cpp` | optional 8th argument: the counter's start value (for the wrap test, §5) |
| `tests/cycle/` (new) | `run_cycle_tests.py` + `programs/*.s`, the test in §5; `make run-cycle-tests` |
| `tests/simt/run_grid_tests.py` | `rdcycle` added to the instructions an SM lane must fault on |

## 4. C helper: `nn_cycles()` (`CRUDA/include/nn.h`)

```c
static inline uint64_t nn_cycles(void);     // host code only
```

It does rdcycleh, rdcycle, rdcycleh and repeats if the two high reads differ, so a read
that straddles the low word's wrap is never torn.

**Overhead.** The helper itself costs cycles. Measured between two back-to-back calls
(host code at `-O2`, the library's convention): **4** cycles on the scalar host and at
width 1, **3** at width 2, **1** at widths 4 and 10. At `-O0` it is 21 / 12 / 10. Against
the lego calls it times (thousands to hundreds of thousands of cycles) this is noise,
and a program can subtract it if it matters.

## 5. Tests

**32I** (`make -C 32I run-cycle-tests`): six hand-written assembly programs, each run on
the scalar host and on the superscalar host at widths 1, 2, 4 and 10 (30 runs). The
expected values are worked out by hand from the rule in §2 and tied to the harness's own
final cycle count:

| Program | Checks |
|---|---|
| `first.s` | `rdcycle` / `rdcycleh` as the first instructions read 0 |
| `same_cycle.s` | two reads back to back differ by 1 (scalar, width 1) or 0 (one group) |
| `total.s` | after a loop with loads, stores and branches, the last read = the harness's final count − the cycles of the last two instructions |
| `launch.s` | the reads around a LAUNCH of one block of 64 threads (2 warps) of `identity.s` differ by 2 × the kernel's instruction count + 2 |
| `wrap.s` | with the counter preset to 2³² − 3: the first reads see high 0 and the low word just below the wrap, the last reads see high 1 and the low word of the final count |
| `system_other.s` | `csrrw` to `cycle`, `csrrs` with `rs1 ≠ x0`, `csrr instret`, ECALL, EBREAK: one error each at the right pc, registers unchanged |

Deliberately broken versions are all caught: `rdcycleh` returning the low half, a read
off by one, `rs1 ≠ x0` accepted, `instret` accepted, `rd` not written (scalar, and
superscalar separately), LAUNCH cycles not added.

**CRUDA** (`run_nn_tests.py --lego cycles`): `tests/t_cycles.c` samples `nn_cycles()` 64
times with the counter preset just below 2³², and checks that each sample is after the
previous one by less than 1,000 cycles and that the count passed 2³². The start value is
swept over 32 consecutive values, so the wrap falls at every point of the sampling loop
(about 20 cycles per sample). A version without the retry fails at the one start value
that puts the wrap between the high and the low read (scalar and width 1 only: the
superscalar host issues both reads in one group, so they can't tear).

**GPU:** `run_grid_tests.py`'s fault test runs `rdcycle` on an SM lane and expects the
`SYSTEM` fault.

## 6. Not included

- **`instret`** (instructions retired). Together with `cycle` it would give the
  superscalar host's IPC from inside a program. Add it when it's needed.
- **A GPU-side clock.** Kernels cannot time themselves; a launch is timed from the host.
- **The lego tests still report whole-program cycles** (start-up and argument setup
  included). Switching `run_nn_tests.py` to report only the lego call, timed with
  `nn_cycles()` inside each `t_<lego>.c`, is a small follow-up.
