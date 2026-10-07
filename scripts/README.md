# scripts/

Two command-line helpers for working with the `32I`/`32IV` emulators. Both are
plain bash scripts, already symlinked into `~/.local/bin`, so they can be run
from anywhere once that's on your `$PATH`.

## `emul` — run a C file on the emulator

```
emul <32I|32IV> <file.c> [max_cycles] [-O0|-O1|-O2|-O3|-Os|-Og|-Ofast] [-mode scalar|superscalar] [-n <issue_width>]
```

Cross-compiles `file.c` bare-metal for RV32IM (using that ISA's
`tests/python/start.s` + `link.ld`), builds that ISA's emulator harness, runs
the binary, and prints the initial and final CPU state (all registers + PC +
cycle count).

- `max_cycles` defaults to `5000` if omitted.
- The optimization level defaults to `-O0` if no `-O*` flag is given;
  `max_cycles` and `-O*` can appear in either order.
- `-mode` defaults to `scalar` (the normal single-instruction pipeline).
  `-mode superscalar` runs it through 32I's in-order superscalar issue path
  instead — see `docs/superscalar.md` for how that works. **32I only**;
  32IV's CPU doesn't have it, and `emul` will refuse the combination.
- `-n <issue_width>` sets the fetch/issue window size for superscalar mode
  (defaults to the CPU's own max, currently 4). Ignored in scalar mode.
- Everything is built in a temp directory that's cleaned up automatically —
  no build artifacts are left behind.

Example:
```
emul 32I main.c
emul 32IV mul_test.c 10000
emul 32I main.c -O3
emul 32I main.c -mode superscalar
emul 32I main.c 10000 -mode superscalar -n 2
```

## `assemblyinstruction` — disassemble, or trace, a C file

```
assemblyinstruction <file.c> [-save [name.txt]] [-trace [name.txt]] [-cycles N] [-O0|-O1|-O2|-O3|-Os|-Og|-Ofast] [-mode scalar|superscalar] [-n <issue_width>] [-gpu full|summary]
```

Compiles `file.c` to RV32IM assembly (`<name>.s`) and an object file
(`<name>.o`) next to the source, then runs
`riscv64-unknown-elf-objdump -d` on the object file and prints the
instructions (address, raw opcode, mnemonic). This is a **static** listing —
every instruction in the binary, once, in link order — regardless of
whether or how many times it actually runs.

- Without `-save`: just prints to the terminal.
- `-save name.txt`: also writes the disassembly to `name.txt` in the same
  folder as the source.
- `-save` with no name: defaults to `<source-basename>.txt`, e.g. `main.c` →
  `main.txt`.
- The optimization level defaults to `-O0` if no `-O*` flag is given;
  `-save` and `-O*` can appear in either order.

`-trace` is different: it cross-compiles the file bare-metal and actually
**runs** it on the 32I emulator, writing a **dynamic** execution trace —
every instruction in the real order it executed, loops repeated as many
times as they really ran (columns: cycle number, PC, disassembled
instruction; custom-0/NPU instructions get an inline `<-- NPU ...` tag) — to
`<name>.trace.txt` (or the given path), instead of the static listing above.
The trace is saved in the directory you run the command from (a relative
path is relative to it too), not next to the source file.

- `-trace` always runs against **32I** (the only tree with the
  halt-on-self-jump idiom the tracer's stop condition relies on) —
  regardless of which ISA the file is meant for.
- `-cycles N` caps how long it'll run before giving up on a program that
  never halts (default `20000000`). It only affects `-trace`.
- A trace can be large — one line per instruction *executed*, not per
  instruction in the binary, so a tight loop run thousands of times produces
  thousands of lines. It's plain tab-separated text, fine to open in an
  editor or `grep`/`less`, just not meant for pasting into a terminal.
- `-mode superscalar` traces 32I's superscalar issue path instead (see
  `docs/superscalar.md`) — the trace format changes: one line per **cycle**
  instead of per instruction (columns: cycle number, base PC, instructions
  issued that cycle, then all of those instructions joined with ` | ` on
  that single line, each still tagged with its own PC and NPU tag). That's
  the point of tracing it this way — a line with more than one instruction
  on it is exactly what `hazard_scan()` decided could co-issue that cycle.
  `-n <issue_width>` sets the fetch/issue window size (defaults to the
  CPU's own max, currently 4). Both are ignored without `-trace`.
- GPU programs (ones that call `simt_launch`, see `32I/tests/simt/simt_isa.h`):
  the GPU's work is printed nested under the host's `LAUNCH` line, then the
  host trace carries on. Each GPU row is `sm  blk  warp  sm_cycle  pc  mask
  instruction` — `sm_cycle` is that SM's own count (SMs run in parallel in
  hardware, so each starts at 0) and `mask` is the set of lanes that ran the
  instruction (watch it narrow after a `SIMT_SPLIT`). Every block ends with a
  `block B on SM S: W warps, N issued` line, and the launch with
  `launch done: device cycles ... (busiest: SM ...)`. `-gpu summary` keeps
  only those per-block lines. SIMT/LAUNCH instructions are shown by name
  instead of objdump's raw `.insn`. With `-mode superscalar` the GPU section
  is nested under the cycle that issued the LAUNCH — a LAUNCH always issues
  alone in its cycle, and the CPU stalls there until the grid is done.

Example:
```
assemblyinstruction main.c
assemblyinstruction main.c -save
assemblyinstruction main.c -save name.txt
assemblyinstruction main.c -O3
assemblyinstruction main.c -save name.txt -O3
assemblyinstruction main.c -trace
assemblyinstruction main.c -trace name.trace.txt -cycles 5000000
assemblyinstruction main.c -trace -mode superscalar
assemblyinstruction main.c -trace -mode superscalar -n 2
assemblyinstruction vec_add.c -trace -gpu summary
```

## Requirements

Both scripts need `riscv64-unknown-elf-gcc` / `-objcopy` / `-objdump` on
`$PATH` (the RISC-V bare-metal toolchain); `emul` and `assemblyinstruction
-trace` also need `g++`.
