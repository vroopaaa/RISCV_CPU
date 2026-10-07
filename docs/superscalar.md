# In-order superscalar issue (32I)

An additive execution mode on `32I`'s `CPU` class, alongside the normal
single-instruction pipeline (`fetch`/`decode`/`execute`/`read`/`writeback`).
It models a classic first-generation superscalar core: in-order issue,
in-order dispatch, in-order writeback, an n-wide fetch window per cycle. No
speculation, no register renaming, no reorder buffer.

The scalar pipeline and the NPU (custom-0, opcode `0x0B`) instructions are
untouched by this — it's a separate set of methods and CPU state, reusing
`decode_one`/`execute_one`/`read_one` (private helpers, one per pipeline
stage, parameterized on a single instruction) so the per-opcode ALU/load/
store/NPU logic isn't written a third time.

## The six methods

Call all six in sequence each cycle, in place of the five scalar-path calls:

```cpp
cpu.fetch_n();      // reads issueWidth (n) consecutive words starting at pc
cpu.decode_all();   // decode_one() + control signals, per slot
cpu.hazard_scan();  // decides m <= n: how many of those n can issue together
cpu.execute_m();    // execute_one() for slots 0..m-1
cpu.read_m();       // read_one() for slots 0..m-1 (loads/stores/NPU)
cpu.writeback_m();  // commits slots 0..m-1 in order, then pc += 4*m
```

`set_issue_width(n)` configures the window size (clamped to
`[1, MAX_ISSUE_WIDTH]`, currently 4). `last_issue_count()` returns `m` from
the most recent `hazard_scan()`.

Unlike a statically-scheduled VLIW bundle, this needs no marker word in the
instruction stream — every cycle it just looks at whatever is next at `pc`.
Whatever doesn't make the cut in `hazard_scan()` becomes the start of next
cycle's window, since `pc` only advances by `4*m`.

## The issue algorithm

`hazard_scan()` scans the decoded window left to right, picking the largest
prefix `m` that's safe to issue together this cycle:

- **LAUNCH issues alone.** A grid launch (custom-2, `0x5B`, see
  `docs/CUDA/plan.md` "Grid launch") cuts the window just before itself and
  comes back as the only instruction of the next cycle, where the CPU stalls
  until the GPU is done. It reads `a0` implicitly (a RAW dependency the
  rs1/rs2 check can't see), it is a full memory barrier, and it can't be
  undone, so it must never sit in a speculative slot -- slot 0 never is.
- **Control hazard.** A branch/`jal`/`jalr` ends the window right after
  itself — `next_pc` isn't known until `execute_m()`, and this codebase does
  no speculation.
- **RAW (true) data hazard.** Instruction `i` is excluded — window truncates
  to `i` — if it reads a register an earlier instruction in the same window
  writes. No forwarding is modeled, so a real dependency just waits for the
  next cycle.
- **WAW.** Not scanned for. `writeback_m()` commits slots `0..m-1` in window
  order, so a later write to the same `rd` correctly overwrites an earlier
  one — the same outcome a real write-port age-priority arbiter would
  produce, just expressed as an ordered commit loop instead of combinational
  mux logic, since this is a functional model, not an RTL one.
- **WAR.** Also not scanned for, and for a more structural reason: every
  slot's register *read* happens in `execute_m()`, and no register is
  actually written until `writeback_m()` runs afterward for the whole
  window. So every read in a cycle sees the same start-of-cycle register
  file regardless of order — the same guarantee a synchronous register file
  gives for free (reads sample pre-edge state, writes land at the clock
  edge). This only holds because the model has no cross-cycle pipeline
  overlap and every instruction has uniform, single-cycle latency; a design
  with multi-cycle ops or deeper pipelining would need to handle WAR for
  real (register renaming, scoreboarding).
- **Memory (structural).** Modeled as a small set of port classes
  (`MemClass` in `CPU.h`: `NONE`/`BANK_A`/`BANK_B`/`STORE`), each standing
  in for a distinct physical resource. Two instructions conflict only if
  they share the *same* non-`NONE` class (`mem_class()`/`hazard_scan()` in
  `CPU.cpp`):
  ```cpp
  if (other_class == my_class) → conflict
  ```
  - Two **stores** — scalar (`0x23`) or the NPU's `store_c` (`0x0B` funct3
    `2`, same class as a plain store) — conflict with each other (single
    external-memory write port).
  - Two loads into the **same** NPU bank conflict (that bank's single write
    port from the NPU side); `load_a` and `load_b` (`0x0B` funct3 `0`/`1`,
    different banks) don't conflict with each other.
  - A **store and an NPU bank load do *not* conflict**, even though both
    have unknown-until-`execute_m()` addresses: a bank load reads external
    memory and writes into the NPU's own internal SRAM, a store writes
    external memory — different classes, modeling separate read/write ports
    rather than one shared bus. (Earlier revisions of this model had them
    conflict, treating "address unknown" as reason enough to serialize
    regardless of direction; this was relaxed once it was clear nothing here
    needs it for correctness — see below — and a separate-port assumption is
    at least as realistic as a shared one.)
  - The debug memory-print instruction (`0x0B` funct3 `3`) is read-only and
    unrelated to the NPU's banks — it never conflicts with anything.
  - Anything with `MemClass::NONE` (ordinary ALU ops, branches handled
    separately above) claims no memory resource at all, so it never
    conflicts with anything, memory-related or not — including a store.

  None of this is required for *correctness* in this model, worth being
  explicit about: `execute_m()`/`read_m()`/`writeback_m()` process every
  slot in a plain sequential loop, so even a genuinely aliasing pair placed
  in the same window would still execute in program order and produce the
  right result. Every rule above exists purely to model plausible hardware
  resource limits, not to keep results correct — which is exactly why they
  keep getting revisited as the "how much hardware are we assuming exists"
  judgment call shifts.

## Running it

See `scripts/README.md`'s `emul` entry for the `-mode superscalar -n
<width>` flags, or invoke `32I/build/run_test` directly:

```
./build/run_test <bin_path> <cycles> 0 0 "" superscalar <issue_width>
```

(The `0 0 ""` are placeholders for the harness's positional memory-dump
args, which sit before `mode`/`issue_width`.) All of `32I/tests/python`'s
existing correctness-suite programs (`mul_test`, `fib_test`, `fact_test`,
`bubble_sort`) produce identical final register state under superscalar
mode at issue widths 1, 2, and 4 as they do under the scalar pipeline.
