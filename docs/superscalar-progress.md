# Superscalar progress log

A chronological record of the in-order superscalar work on `32I`: what was
built, what the hazard model went through, what tooling came out of it, and
what the actual measured numbers were at each stage. `docs/superscalar.md`
is the up-to-date design reference (current rules only); this file is the
story of how it got there, kept so the reasoning and the dead ends aren't
lost.

## 1. Core model

Built from scratch as an additive path alongside the existing scalar
`fetch`/`decode`/`execute`/`read`/`writeback` pipeline (untouched by any of
this): in-order issue, in-order dispatch, in-order writeback, an n-wide
fetch window per cycle, no register renaming, no reorder buffer, no
speculation (later revisited — see §6). Six new `CPU` methods —
`fetch_n`/`decode_all`/`hazard_scan`/`execute_m`/`read_m`/`writeback_m` —
reuse `decode_one`/`execute_one`/`read_one` (private per-instruction
helpers) so the per-opcode ALU/load/store/NPU logic isn't written a third
time.

`hazard_scan()` picks the largest hazard-free prefix of the window each
cycle:
- **Control** — a branch/`jal`/`jalr` ends the window right after itself
  (no speculation).
- **RAW** — instruction `i` excluded if it reads a register an earlier
  slot in the window writes.
- **WAW** — no scan needed: `writeback_m()` commits `0..m-1` in order, so a
  later write to the same `rd` naturally overwrites an earlier one.
- **WAR** — no scan needed: every slot's register *read* happens in
  `execute_m()`, before any slot's write lands in `writeback_m()`.
- **Memory (structural)** — see §3, this is the piece that got revised the
  most.

Verified against `32I/tests/python`'s real correctness suite
(`mul_test`/`fib_test`/`fact_test`/`bubble_sort`) at every stage — always
bit-identical final register state to the scalar pipeline, at every issue
width tried.

## 2. Cleaning up the ground first

Before any of this, a stray uncommitted "VLIW bundle" prototype (opcode
`0x2B` marker words, static offline scheduling) was sitting in the working
tree, unrelated to what was being built here. Confirmed the real
`vliw-prototype` branch only touched NPU test/Python files, not
`CPU.h`/`CPU.cpp` — the CPU-side bundle code was pure uncommitted
scratch work, never recoverable from that branch. Discarded it
(`git checkout -- CPU.h CPU.cpp`) back to the last commit, then built the
superscalar path fresh on the clean base.

## 3. The memory-hazard model's evolution

This went through three real iterations, each one caught by actually
looking at generated traces, not by reasoning in the abstract.

**v1 (initial).** A store — scalar `SW` or the NPU's `store_c` — closed the
*entire* window unconditionally the moment it was seen, regardless of what
followed. NPU `load_a`/`load_b` (into separate banks) could co-issue with
each other. This was deliberately conservative ("for realism, cheap and
safe" — modeling a single memory port).

**v2 (store no longer blocks unrelated ALU ops).** Looking at a real
matmul trace showed `sw a1,0(a6)` (the NPU MAC trigger) followed by
`addi a5,a5,1024` / `add a4,a4,a0` — both pure address-increment ALU ops,
zero relation to memory — landing in *separate* cycles for no real reason.
v1's forward closure didn't check whether what followed even touched
memory; it excluded everything. Replaced with a proper `MemClass`
(`NONE`/`BANK_A`/`BANK_B`/`STORE`) pairwise check: a store now only
conflicts with something that shares a real memory-port resource. Anything
with `MemClass::NONE` is free to co-issue with a store in either position.
Verified: `store-vs-store` and `same-bank-twice` conflicts still correctly
caught; `load_a`+`load_b`+`sw`+`addi` now co-issue as one group when
otherwise independent.

**v3 (store and an NPU bank load are different resources).** Further
trace inspection showed `load_a`/`load_b` still never co-issuing with a
`store` even after v2, because the check still treated "store" and
"either NPU bank" as the same contended resource (`my_class == MEM_STORE
|| other_class == MEM_STORE`). Reasoned through it: a bank load reads
external memory and writes into the NPU's own internal SRAM; a store
writes external memory — different physical ports (read vs. write), not
actually the same resource. Simplified the whole check to one line —
two instructions conflict only if they share the exact same `MemClass`:
```cpp
if (other_class == my_class) { mem_conflict = true; break; }
```
This is strictly simpler code than v2 *and* v1, and it's what's live today.
Re-verified `store-vs-store`/`same-bank-twice` still excluded, and the
full NPU matmul correctness suite (see §4) still bit-exact at every issue
width tested.

A note that came out of all three rounds, worth keeping: **none of these
memory rules are needed for correctness** in this model.
`execute_m()`/`read_m()`/`writeback_m()` all process slots in a plain
sequential loop, so even a genuinely aliasing pair placed in the same
window would still execute in program order and produce the right result.
Every one of these rules exists purely to model plausible hardware
resource limits — which is exactly why they kept getting revisited as the
"how much hardware do we assume exists" judgment call shifted.

## 4. Tooling built

- **`tests/basic/main.cpp`**: two new trailing, fully optional args —
  `mode` (`scalar`/`superscalar`) and `issue_width` — backward compatible
  with every existing caller.
- **`scripts/emul`**: `-mode scalar|superscalar` / `-n <width>`, guarded
  against `-mode superscalar` on `32IV` (no superscalar CPU there).
- **`scripts/assemblyinstruction -trace`**: same `-mode`/`-n` flags; in
  superscalar mode the trace format changes to one line per *cycle*
  (`cycle, base_pc, issued, instructions`), with everything issued
  together joined by ` | ` — the whole point being to see co-issue
  groupings directly. Along the way, found and fixed a **pre-existing
  break**: the tracer's NPU-tagging accessors (`last_opcode()`/
  `last_funct3()`) had been silently dropped during the VLIW cleanup in
  §2; restored them plus new `issued_opcode(slot)`/`issued_funct3(slot)`
  for the superscalar window.
- **`tests/npu/run_npu_tests.py`**: added `--mode`/`--issue-width`/`--opt`
  (compiler optimization level, previously hardcoded to `-O0`), plus
  cycle-count reporting (`run_simulator()` now returns the harness's own
  `Cycles: N`). This is what caught that the superscalar path had *never*
  actually been run through the real NPU matmul correctness check before —
  only synthetic register-level hazard tests had exercised it.
- **`tests/npu/run_npu_tests_2.py`** (new): same correctness check, but
  with the K-tile loop unrolled by a configurable factor before
  compilation (`--unroll 0` = fully flatten, `--unroll N` = partial with a
  remainder loop). Reuses `run_npu_tests.py`'s machinery via import.
  Companion template: `c_tests/matmul_template_2_unrolled.c`.
- **`tests/npu/analyze_superscalar.py`** (new): automates the
  compile-once-per-opt-level, trace-scalar-and-n=1..N sweep, computing
  cycles/avg-IPC/speedup/co-issue distribution per level, with an optional
  JSON dump.

## 5. Correctness verification

The NPU matmul correctness suite (cross-compile → run on emulator → diff
against an independent Python matmul) was run, bit-exact, across:
- Both templates (1: per-word MMIO: 2: custom-0 tile-transfer).
- Sizes 16×16, 32×32, 256×256.
- Issue widths 1, 2, 3, 4, 6, and 10 (the CPU's configured max).
- Optimization levels O0–O3.
- Multiple random seeds.
- The fully-unrolled and partially-unrolled K-loop variants.
- Every stage of the memory-hazard-model evolution in §3.

No mismatch was ever found. This matters more than it sounds: it's the
actual proof that letting `load_a`/`load_b` co-issue, letting stores ride
with independent ALU ops, and letting bank loads and stores co-issue, all
still produce a numerically correct matmul — not just that the hazard scan
*looks* safe in the abstract.

## 6. Performance results

All numbers below are for the 256×256 tiled NPU matmul
(`matmul_256_t2.c`/`matmul_256_t2u_*`, seed 42 unless noted), template 2.

### 6a. Issue-width convergence (measured right after the core model, v1
memory rules, before the §3 relaxations)

```
Level   Total instr   n=1      n=2      n=3      n=4      n=5      n=6      Best speedup
O0      291,288       291,288  210,351  201,900  196,524  188,331  188,331  1.547x (n=5)
O1       27,299        27,299   18,513   18,508   18,489   18,489   18,488  1.477x (n=6)
O2       27,537        27,537   18,508   18,487   18,483   18,481   18,481  1.490x (n=5)
O3       23,263        23,263   18,318   18,249   18,229   18,229   18,228  1.276x (n=6)
```

Key finding at this stage: total instruction count is invariant across
issue width (only cycles change) — confirmed exactly at every level.
Almost the entire achievable speedup happens going from n=1 to n=2;
everything past that is a rounding error (n=5→n=6 identical at O0/O2).
Distribution: at every level, ~50% of cycles issue exactly 1 instruction,
~45% issue exactly 2, and 3+ is under 2% of cycles — a two-bucket story,
not a wide spread.

### 6b. After the §3 v2 relaxation (store co-issues with independent ALU)

```
Level   Best speedup (v1 rule)   Best speedup (v2 rule)
O0            1.547x                   1.927x
O1            1.477x                   1.967x
O2            1.490x                   1.985x
O3            1.276x                   1.374x
```

Notable flip: under v1, **O0** had the best speedup (unoptimized code has
more redundant independent loads/stores lying around, easy to co-issue).
Under v2, **O2** overtakes it (1.985x) — once stores can also ride with
unrelated ALU ops, O2's tighter-but-still-somewhat-redundant code exploits
that better than O0's raw redundancy does. O3 barely moves either way
(already-tight scheduling leaves little slack regardless of the rule).

### 6c. After the §3 v3 relaxation (store and NPU bank load are different
resources) — current rules, current numbers, 256×256 O2

```
                              scalar     n=6 (superscalar)
regular (not unrolled)        26,763           9,023   (avg IPC ~2.97)
```
(This O2 number is reproduced exactly by the full re-sweep in §6d below —
same seed, same current rules.)

### 6d. Full O0–O3 × n=1..10 sweep (current rules: v3 memory model + §7
reset optimization), via `analyze_superscalar.py --max-n 10`

256×256, template 2, seed 42. Total instruction count is invariant across
`n` for a fixed level (confirmed again here), and all 44 runs (4 levels ×
[scalar + n=1..10]) halted normally.

```
--- O0  (total instructions = 279,303) ---
   n        cycles    avg IPC    speedup
scalar      279,303     1.0000     1.000x
   1        279,303     1.0000     1.000x
   2        180,664     1.5460     1.546x
   3        158,877     1.7580     1.758x
   4        158,876     1.7580     1.758x
   5        146,072     1.9121     1.912x
   6..10    146,072     1.9121     1.912x   (converged at n=5)

--- O1  (total instructions = 26,525) ---
   n        cycles    avg IPC    speedup
scalar       26,525     1.0000     1.000x
   1         26,525     1.0000     1.000x
   2         13,388     1.9813     1.981x
   3         13,128     2.0205     2.020x
   4          9,028     2.9381     2.938x
   5          9,028     2.9381     2.938x
   6          9,028     2.9381     2.938x
   7          9,027     2.9384     2.938x
   8-10       9,027     2.9384     2.938x

--- O2  (total instructions = 26,763) ---
   n        cycles    avg IPC    speedup
scalar       26,763     1.0000     1.000x
   1         26,763     1.0000     1.000x
   2         13,641     1.9620     1.962x
   3         13,123     2.0394     2.039x
   4          9,026     2.9651     2.965x
   5          9,025     2.9654     2.965x
   6          9,023     2.9661     2.966x
   7          9,023     2.9661     2.966x
   8-10       9,022     2.9664     2.966x   (matches §6c/§7's n=6 figure
                                              of 9,023 almost exactly --
                                              1 cycle better at n=8+)

--- O3  (total instructions = 21,552) ---
   n        cycles    avg IPC    speedup
scalar       21,552     1.0000     1.000x
   1         21,552     1.0000     1.000x
   2         12,488     1.7258     1.726x
   3         11,358     1.8975     1.898x
   4          8,267     2.6070     2.607x
   5          8,266     2.6073     2.607x
   6-7        8,265     2.6076     2.608x
   8-10       8,264     2.6079     2.608x

--- Cross-level summary (best speedup, n=1..10) ---
 level   total instr   best speedup
    O0       279,303         1.912x
    O1        26,525         2.938x
    O2        26,763         2.966x
    O3        21,552         2.608x
```

Findings, comparing to §6a/§6b (pre-v3, pre-reset-optimization numbers):
- **Instruction counts dropped at every level**, not just O2 — the §7
  reset optimization (dims hoisted out of the tile loop) removes 3
  `npu_write32` calls per tile regardless of optimization level, so O0/O1/
  O3 all show fewer total instructions than their §6a figures too (e.g. O0
  291,288 → 279,303; O3 23,263 → 21,552). O2's count (26,763) is unchanged
  from §7, confirming that measurement already reflected current rules.
- **The n=10 ceiling barely moves anything past n=4-6.** O0 is flat from
  n=5 on; O1/O2/O3 each shed one or two more cycles at n=7-8 and then hold
  flat through n=10 — a sub-0.1% tail, not a real trend. Widening past n=6
  buys effectively nothing at any level, confirming §6a's "almost the
  entire speedup happens by n=2, the rest is a rounding error" finding
  still holds at the wider n=10 ceiling.
- **O2 remains the best overall** (2.966x), same ranking as §6b (v2
  numbers), now with the v3 + reset-optimization rules. O3's speedup is
  still the weakest of the four (2.608x) — already-tight scheduling at
  -O3 leaves the least slack for co-issue, consistent with every prior
  measurement in this log.
- Raw traces for all 44 runs and the machine-readable report are saved
  under `32I/tests/npu/build/` (`superscalar_sweep_n10.json` plus one
  `matmul_256_t2_<OPT>_<label>.txt` trace per run).

## 7. NPU `reset()` optimization

`reset()` used to clear `M`/`K`/`N` (the NPU's configured dimensions)
along with `A`/`B`/`C`/`done` — meaning every template had to re-declare
`DIM_M/K/N_ADDR` after *every single tile's* reset, even though for this
tiled matmul every tile is 16×16 and the dims never actually change across
the whole run. Changed `reset()` to leave `M`/`K`/`N` alone (only `A`/`B`/`C`
and `done` are cleared), and hoisted the `DIM_*_ADDR` writes above the tile
loop in all three templates (`matmul_template.c`, `matmul_template_2.c`,
`matmul_template_2_unrolled.c`) — set once instead of once per tile (256
times for a 256×256 test). Backward compatible: templates that still
re-declare dims every tile (redundant now) remain correct.

Measured effect, O2, 256×256, regular (not unrolled) template:
```
                    scalar cycles
before (dims/tile)     27,537
after (dims once)      26,763     (774 fewer, ~1:1 with the 765 removed npu_write32 calls)
```

## 8. Loop-unrolling experiments

Motivation: every K-loop iteration ends in a branch (`bne`), and a branch
*always* closes the issue window regardless of `n` — the one hazard type a
wider window structurally cannot route around. Built `run_npu_tests_2.py`
+ the unrolled template (§4) to flatten the K-loop and test the
hypothesis.

**First attempt — literal `Kt` per unrolled copy.** Correct results, real
but modest gains:
```
                    scalar      n=4       n=10
not unrolled       291,288    164,742   151,173   (O0, before dims-hoist)
unrolled            252,632    140,934   127,365
```
~13-19% cuts, not the "~half" a naive branch-removal estimate would
suggest.

**Investigating why it wasn't bigger** (at O2, post dims-hoist,
non-unrolled 26,763 / unrolled 21,712 scalar — an 18.9% cut): disassembled
both. The K-loop body at `-O2` is only 6 instructions
(`load_a`/`load_b`/`sw`(MAC trigger)/2 address-increment adds/`bne`) — the
branch is 1 of 6, so removing it caps the *scalar* win at roughly 1/6 of
the loop's cost, which is what was measured.

**But the superscalar picture (`-n 6`) told a sharper story.** The
non-unrolled trace showed a clean, repeating pattern: `load_a`+`load_b`+
`sw`+`addi`+`add` co-issue as one 5-wide group, then `bne` executes
*completely alone* in the next cycle (nothing joins it, since a branch
that becomes slot 0 of a fresh window immediately closes that window —
no speculation past it). That "alone in a 6-wide-capable window" pattern
predicts a much bigger win from removing it (~4096 cycles across the
whole K-loop) than what unrolling actually delivered:
```
                    n=6 cycles
not unrolled          9,023
fully unrolled        8,345    (only 678 fewer — not ~4096)
```

**Root cause, found by tracing the actual unrolled binary**: with `Kt` as
a compile-time literal per copy, `-O2` stopped incrementing a running
pointer (the loop's cheap `addi a5,a5,1024`) and instead rebuilt each
tile's *absolute* address from scratch — a `lui`+`add` pair, where `add`
directly reads the register `lui` just wrote. That's a new RAW hazard,
forcing its own lone cycle, in the exact same "alone in a wide window"
shape that `bne` had. Net effect: removing one bottleneck (the branch)
introduced a different one (the address rebuild) of almost the same size.

**Attempted fix #1 — incremental C-level pointers** (`pA += TILE_WORDS`
instead of recomputing `&A[(I*TILES+Kt)*TILE_WORDS]`): **had zero effect**.
`-O2`'s optimizer normalizes both source forms back to the same generated
code once everything is a compile-time constant — the disassembly was
identical either way (still 8,345 cycles).

**Attempted fix #2 — partial unroll (factor 4), keeping `Kt` a genuine
runtime variable**: this *did* change the generated code (GCC precomputed
stride multiples once, outside the block, avoiding `lui` inside the loop
entirely) — but the result was **worse**, not better:
```
partial unroll (factor 4), n=6:  9,032 cycles   (vs. 9,023 not unrolled)
```
Each load's address is still computed by an `add` immediately before the
load that consumes it — same RAW shape, just a cheaper producer
instruction — and partial unroll keeps a `bne` running `TILES/unroll`
times per tile instead of removing it, clawing back part of what
unrolling was supposed to save.

**Conclusion (current end state of this thread)**: ~8,300–9,000 cycles at
`-n 6` for this workload appears to be close to the real floor under the
current no-forwarding model — not because of the branch specifically, but
because `load_a`/`load_b`/`store_c` all take a register argument, and
*something* has to compute that register's value in the instruction right
before it, which can never co-issue with the instruction consuming it.
Unrolling changes which instruction plays that role; it doesn't remove
the role. Getting materially past this floor would need either register
forwarding (a bigger model change) or software pipelining (computing
address *N+1* several instructions ahead of when it's needed, so
something independent fills the gap) — not more unrolling.

## 9. Open threads / natural next steps

- ~~A fresh full O0–O3 × n=1..6 sweep hasn't been re-run since the §3 v3
  relaxation + §7 reset optimization~~ — done, see §6d (extended to
  n=1..10 while at it).
- **Branch speculation was designed but not implemented.** Two variants
  discussed:
  - *Predict-not-taken with squash*: keep fetching sequentially past a
    branch (which is already what `fetch_n()` does — the fall-through
    path is just "what's physically next"), execute speculatively, and
    squash (skip `read_m()`/`writeback_m()` effects for) everything after
    it if `execute_m()` finds it was actually taken. Cleanly buildable on
    the current pipeline ordering (branches resolve in `execute_m()`,
    before any real side effect happens in `read_m()`) — but predicting
    **not-taken is the wrong guess for backward (loop-closing) branches**,
    which dominate this workload (taken on 15/16 K-iterations). Would
    provide close to zero benefit for exactly the code this session
    focused on.
  - *Predict-taken for backward branches*: more useful for this workload,
    more invasive to build. Since RISC-V branch targets are PC-relative
    immediates known at decode time (no real branch-target-buffer needed),
    `decode_all()` can redirect the fetch address for the *rest of the
    current window* the moment it decodes a backward branch. Requires:
    a per-slot PC array (`windowSlotPC[]`, since slot address is no longer
    simply `windowBasePC + 4*i` once a mid-window redirect can happen),
    `fetch_n()`/`decode_all()` merged into one interleaved pass, and a
    squash path for the misprediction direction (predicted taken, actually
    not taken → redirect to the branch's own fall-through instead).
    Designed in detail (see conversation), not yet implemented.
- **Real forwarding or software pipelining** as the path past the
  ~8,300-cycle floor identified in §8, if that's ever worth pursuing.
