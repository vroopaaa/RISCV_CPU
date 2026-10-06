# simulation/

A throwaway Python prototype of the SIMT execution model (warps, lanes,
masking, shared memory, lockstep broadcast) — built to validate the
*semantics* before porting them into the C++ emulator's `SIMTCore`
(`32I/include/SIMT.h`). No RISC-V encoding, no compiler, no `.insn` here —
kernels are plain Python lists of symbolic ops.

## Files

- `simt_model.py` — the model itself: `Warp` (one PC + tmask + per-lane
  registers) and `SIMTCore` (shared memory + the warp + the fetch/broadcast
  cycle loop + one handler per op).
- `kernels.py` — kernels as data: lists of symbolic ops, e.g. the
  scalar-multiply-by-4 example from the CUDA reference.
- `run_scalar_mul.py` — driver: seeds shared memory, runs a kernel to
  completion, checks the result, optionally prints a per-cycle trace.

## Instruction format

Each op is a tuple `(MNEMONIC, *args)`. Registers and shared-memory
addresses are just Python ints (lane-local register index / flat list
index) — no encoding, no immediates-vs-registers distinction unless you
want one.

Planned op set (fill in as needed, not all of these have to exist from
day one):
- `("TID",)` — write this lane's index into a fixed "idx register".
- `("LOADI", reg, value)` — load an immediate into a register.
- `("LOAD", dst_reg, addr_reg)` — `regs[dst_reg] = shared_mem[regs[addr_reg]]`.
- `("STORE", addr_reg, src_reg)` — `shared_mem[regs[addr_reg]] = regs[src_reg]`.
- `("MUL", dst_reg, src1_reg, src2_reg)`.
- `("ADD", dst_reg, src1_reg, src2_reg)`.
- `("BRANCH", cond_reg, target_pc)` — lockstep: for now, all active lanes
  must agree on the branch outcome (no divergence handling yet).
- `("HALT",)` — warp retires.

## Running

```
python3 run_scalar_mul.py
```
