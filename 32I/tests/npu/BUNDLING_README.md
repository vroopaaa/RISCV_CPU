# VLIW-style bundling of the NPU matmul Kt-loop — progress notes

Status as of this session: **core mechanism works and is proven correct for
BIG_DIM 64/128/256; BIG_DIM 512 is a known-broken edge case; one pending
code revert; verification harness (`run_bundle_tests.py`) not yet written.**

Full design context/history is in the conversation this was built in — this
file is a resumption point, not a replacement for that.

## What's done

### 1. CPU-side bundle execution (`32I/include/CPU.h`, `32I/src/CPU.cpp`)

Purely **additive** — the original single-instruction pipeline
(`instruction`/`aluResult`/`memResult`/`decodedInstruction` and the bodies of
`fetch()`/`decode()`/`execute()`/`read()`/`writeback()`) is untouched,
byte-for-byte, for the non-bundled path. New, parallel state added to
`CPU.h`: `bundleActive`, `bundleSlotCount`, `slotBasePC`, `rawSlotWords[]`,
`decodedSlots[]`, `slotMemReadEnable[]`/`slotMemWriteEnable[]`/
`slotRegWriteEnable[]`, `slotAluResult[]`/`slotMemResult[]`
(`MAX_BUNDLE_SLOTS = 4`), plus new private helpers `decode_one`/
`execute_one`/`read_one` (duplicated per-opcode logic, not shared with the
original switches, by design — see the "old path fully untouched" decision
below).

**Marker recognition lives in `decode()`, not `fetch()`.** `fetch()` still
just reads the one word at `pc` into `instruction`, exactly as before —
deciding "this word means more words follow" is interpreting the
instruction's meaning, which is decode's job. `decode()` checks
`(instruction & 0x7F) == 0x2B`; if so, it reads `funct7` for `K` (real
instruction count), reads the `K` words after the marker directly from
memory itself, decodes each via `decode_one`, and overrides `next_pc`.
`execute()`/`read()`/`writeback()` each have a `bundleActive` guard at the
top that loops over the bundle's slots using `execute_one`/`read_one`/staged
register writes, then fall through to the original body when not bundled.

**Marker encoding**: single opcode `0x2B` for every bundle (not two, as an
earlier draft in this session had — see conversation history for why that
was reverted). The marker is a **genuinely extra word**, not a replaced
instruction: `opcode=0x2B`, `funct7=K` (count of real instruction words
immediately following), `rd`/`funct3`/`rs1`/`rs2` unused/0. A bundle on disk
is `marker(K), instr_0, ..., instr_{K-1}` — `K+1` words, `K` real
instructions, one `cycle_count++` regardless of `K`.

**Verified**: `run_tests.py`, `run_opt_tests.py`, and `run_npu_tests.py`
(the existing suites, none touching bundling) all still pass, cycle counts
unchanged — confirms the additive change didn't disturb the non-bundled
path.

**Known temporary debug instrumentation still in the file** (needs
removing before this is considered done): both `read()`'s and `read_one`'s
`opcode == 0x0B` blocks in `CPU.cpp` have a
`if (getenv("BUNDLE_DEBUG")) fprintf(stderr, ...)` line printing the NPU
tile-transfer's `funct3`/base address. Added to diagnose the relocation bug
below; harmless (gated, off by default) but should be deleted once bundling
work is finished.

### 2. Python harness (`32I/tests/npu/bundle_kt_loop.py`, new file)

Does its own `-O2` compile (reusing `run_npu_tests.py`'s
`generate_matrix`/`render_test_c` as a library) since `run_npu_tests.py`
itself hardcodes `-O0`, which never produces the inlined idiom at all.

Pipeline: `disassemble()` (objdump → `Insn` objects, fields decoded from raw
hex the same way `decode_one` does) → `find_bundle_windows()` (structural +
register + `track_constants`-verified match of the 6-instruction
`LOAD_A, LOAD_B, TRIG, INC_A, INC_B, BR` idiom, in that real compiled
program order — note TRIG sits *between* the loads and the increments, not
after, so bundling requires reordering, not just tagging) → `bundle_match()`
(reorders into `marker(K=4), LOAD_A, LOAD_B, INC_A, INC_B, marker(K=2),
TRIG, BR`, relocates branches/jumps whose source and target straddle the
+8-byte insertion point, relocates absolute-address constants referencing
`&A[...]`/`&B[...]`).

**The relocation-worthy bug found and fixed this session**: RISC-V linker
relaxation collapses a `lui+addi` pair down to a single bare
`addi rd,x0,imm` whenever the final linked address's upper 20 bits are zero
— true for whichever of `A`/`B` lands first in `.rodata`, right after a
small `.text` section. `track_constants()` originally only recognized the
two-instruction form, silently missing this and corrupting `B`'s reads for
some sizes. Fixed by also tracking bare `li` as a constant-materializing
candidate, and grounding *which* constants are actually relocation-eligible
in the ELF's real symbol table (`symbol_addresses()`, via `nm`) rather than
a numeric-range guess — a range guess risks false positives (an ordinary
integer like `BIG_DIM` can coincidentally fall in a plausible-looking
address range).

## What's verified (empirically, via the sweep below)

| Size | TILES | Bundled? | Cycles (unbundled → bundled) | Matches `unbundled − 4·TILES³`? | Bit-exact vs. Python reference? |
|---|---|---|---|---|---|
| 16 | 1 | No match (GCC fully unrolls a 1-iteration loop) | 29 | — | PASS |
| 32 | 2 | No match (GCC fully unrolls a 2-iteration loop) | 85 | — | PASS |
| 48 | 3 | No match | 323 | — | PASS |
| 64 | 4 | **Yes** | 636 → 380 | YES | **PASS** |
| 80 | 5 | No match | 1113 | — | PASS |
| 96 | 6 | No match | 1784 | — | PASS |
| 112 | 7 | No match | 2699 | — | PASS |
| 128 | 8 | **Yes** | 3887 → 1839 | YES | **PASS** |
| 256 | 16 | **Yes** | 27016 → 10632 | YES | **PASS** |
| 512 | 32 | **Yes** | 206163 → ??? | — | **FAIL** (see below) |

(GCC only unrolls the Kt-loop away for small TILES — the pattern-matcher
correctly finds nothing to bundle in that case, which is exactly the
16/32/48/80/96/112 rows above; the *user's own prediction*, made before
running anything, was specifically "32×32 won't change" — confirmed.)

Sweep script (throwaway, reusable to re-verify after further changes):
`/tmp/claude-1000/-home-roopi-Desktop-rsvp/2e0f738e-0b92-4021-8840-7645f73a2faf/scratchpad/bundle_size_sweep.py`
— not in the repo, may not survive session cleanup; rewrite if gone (it's
short: loop over sizes, call `bundle_kt_loop`'s `build_candidate`/
`find_bundle_windows`/`bundle_match`, run both binaries through
`build/run_test`, diff against a numpy-computed reference).

## The 512×512 problem (open)

GCC's `-O2` output for `BIG_DIM=512` uses a **third address-computation
idiom** beyond the two `track_constants()` handles: something like
`lui s6,0xfff38` (a large *negative*-looking constant) later combined via
`sub`/`add` rather than a same-register completing `addi`. That constant's
final value is never captured by `track_constants()`, so if it's actually
address-relevant, it silently goes unrelocated.

**A static "reject any unresolved `lui`" safety check was tried and
reverted in this session** — it was too blunt: most dangling `lui`s are
ordinary stride constants (e.g. a row-stride added to an *already correctly
relocated* base elsewhere), not addresses needing relocation themselves,
and the check ended up rejecting 64/128/256 too, which are proven correct
by actually running them. **Pending as of the interruption that led to this
README: the edit reverting that check was proposed but not yet applied** —
`bundle_match()` in `bundle_kt_loop.py` may still contain the
`if unresolved_luis: raise BundlingUnsupported(...)` block. Check this
first when resuming; if present, decide whether to remove it (recommended —
see next section) or replace it with something better-targeted.

## Recommended next steps (not yet decided/started)

1. **Resolve the pending revert**: remove (or replace with something more
   targeted than "any dangling lui") the static safety check in
   `bundle_match()`, since it currently blocks the three sizes already
   proven correct.
2. **Decide how to treat 512×512 (and any other size that fails)**: the
   project's actual verification philosophy throughout (Kt-loop matching
   itself, the whole NPU test suite) is *run it and diff against a
   reference*, not prove safety statically — so the natural fit is:
   `run_bundle_tests.py` (see below) runs the bundled binary, diffs against
   an independent reference, and **that's** what determines whether a size
   is "supported" — not a compile-time guess. A size that fails just gets
   reported as failing (and, per the earlier user-approved direction,
   presumably shouldn't be shipped/relied on) rather than the tool trying to
   silently guess it's safe.
3. **Write `run_bundle_tests.py`** (per the original approved plan, not yet
   started): for each requested size, build unbundled, run
   `bundle_kt_loop.bundle_binary`, run both through the emulator, assert
   bit-exact vs. an independent Python/numpy reference, assert the cycle
   count dropped by the predicted amount. This automates what the throwaway
   sweep script above does by hand.
4. **Remove the `BUNDLE_DEBUG` instrumentation** from `CPU.cpp` once
   bundling work is finished (two `fprintf(stderr, ...)` lines, both gated
   behind `getenv("BUNDLE_DEBUG")`, in `read()` and `read_one()`'s
   `opcode == 0x0B` handling).
5. Only after the above: consider whether/how to document the
   64–256 (and not 512) supported range somewhere durable (this file, or a
   comment in `bundle_kt_loop.py`'s module docstring already says
   "verified/supported range: BIG_DIM 64-256" in a couple of places —
   worth double-checking those stay accurate once 512 is actually resolved
   one way or the other).

## Files touched this session

- `32I/include/CPU.h`, `32I/src/CPU.cpp` — additive bundling support (see
  above). `git diff` these to review; nothing pre-existing was removed.
- `32I/tests/npu/bundle_kt_loop.py` — new, the pattern-matcher/patcher
  described above.
- `32I/tests/npu/run_bundle_tests.py` — **not yet created**.
- This file.
