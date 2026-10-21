# NPU bias preload and output stage (32I hardware change)

Two additions to the NPU so a matrix multiply can hand the next lego an int8 tensor
directly:

- **Bias preload.** A tile's accumulator `C` can start from the bias (one int32 per
  output column, broadcast down every row) instead of zero. The MACs then add on top.
- **Output stage.** After the MAC array: rounding right shift and saturation to int8,
  written straight to memory as one tile-major int8 tile.

Before this change the NPU could only store its int32 accumulator (`store_c`), so a
GEMM's result needed a separate pass (bias, shift, clamp) to become int8. Real
accelerators do this next to the array: NVDLA's SDP, the TPU's activation pipeline,
Ascend's vector unit, and accumulator bias preload in many designs.

Everything that existed before keeps working unchanged: `load_a`, `load_b`, `store_c`,
the memory print, the MMIO registers, `compute`/`mac`/`reset`, and every existing NPU
test.

## 1. What it computes

A tile is computed as:

```
C(i, j) = bias[j]        for every row i     (load_bias)    -- or 0 after RESET
C      += A · B          once per K tile     (MAC)
out(i, j) = sat8(rshift(C(i, j), shift))     if i < M and j < N     (store_c8)
          = 0                                 otherwise
```

- `rshift` and `sat8` are exactly the rules in [00-overview.md](00-overview.md) §1
  (round half toward +∞, clamp to [-128, 127]).
- `M` and `N` are the NPU's existing `DIM_M` / `DIM_N` registers. Elements outside the
  `M × N` region are written as 0. This keeps the library's zero-padding rule: at the
  bottom or right edge of a matrix, the host sets `M` / `N` to the real number of rows
  and columns in that tile, and the padding comes out as 0 instead of `bias >> shift`.

### Why the bias goes into the accumulator

The bias must be added before the shift and clamp. Adding it afterwards changes the
result: a large accumulator clamps to 127 before a negative bias can pull it back, and
a pre-shifted bias loses its low bits. So the bias has to enter at int32, at the
accumulator. Preloading it into `C` does that with **no extra storage and no extra
adders**: `C` and its adders already exist. It also leaves no state behind, since each
tile starts fresh from either `load_bias` or `RESET`.

An earlier version of this design kept the bias in a separate 16-entry register that
the output stage added. It was replaced by the preload: same result, 512 fewer bits of
storage and 16 fewer adders, and no bias that silently carries over to the next GEMM.

## 2. Instructions (custom-0, opcode `0x0B`)

Same encoding as the existing ones: `.insn r 0x0B, funct3, 0, x0, rs1, rs2`.

| funct3 | Name | rs1 | rs2 | Effect |
|---|---|---|---|---|
| 0 | `load_a` | src | — | *(existing)* |
| 1 | `load_b` | src | — | *(existing)* |
| 2 | `store_c` | dst | row stride (bytes) | *(existing)* int32 tile, strided |
| 3 | memory print | src | N | *(existing)* debug |
| **4** | **`load_bias`** | src | — | read 16 contiguous int32 (64 bytes); set `C(i, j) = bias[j]` for all 16 rows. Replaces `RESET` at the start of a tile. |
| **5** | **`store_c8`** | dst | shift | run the output stage on `C` and write the 256-byte int8 tile contiguously at `dst` |

- `store_c8` uses the low 5 bits of `rs2` as `shift` (0 – 31).
- `store_c8` writes all 256 bytes, padding included, so the destination tile is always
  fully defined. Its layout is the library's tile layout: row-major inside the tile,
  `out(i, j)` at byte `i·16 + j`.
- A tile with no bias starts with `RESET` (C = 0) as before.
- Both instructions cost one instruction (one cycle), like the existing tile transfers.
  This models a 256-byte-per-cycle path between the NPU and its memory, which is
  realistic for on-chip SRAM. See §7.

## 3. Memory-mapped view (`NPU.h`)

| Address | Size | Access | Contents |
|---|---|---|---|
| `OUT_SHIFT_ADDR` = `NPU_BASE + 0x20` | 1 word | read/write | output-stage shift (low 5 bits). Configuration like `M/K/N`: `RESET` keeps it. |
| `BIAS_ADDR` = `RESULT_ADDR + TILE_ELEMS·4` | 16 words | **write-only** | bias preload port: writing word `j` sets column `j` of `C` to that value in every row. No storage behind it. |
| `OUT8_ADDR` = `BIAS_ADDR + 64` | 256 bytes | read-only | the output stage's int8 result, computed on read from the current `C`, `M`, `N` and shift |

`NPU_WINDOW_SIZE` grows to include them, so `Memory` routes them to the NPU with no
change to `memory.cpp`. The instructions use this view the same way `load_a` and
`store_c` use theirs:

- `load_bias`: 16 word writes into `BIAS_ADDR`.
- `store_c8`: one write of `shift` to `OUT_SHIFT_ADDR`, then 64 word reads from
  `OUT8_ADDR`, each written to `dst`.

A host program can also drive both through MMIO alone (write the bias port and the
shift, read `OUT8_ADDR`). The instructions are the fast path, as with the tile loads.

## 4. Changes in 32I

| File | Change |
|---|---|
| `include/NPU.h` | `OUT_SHIFT_ADDR`, `BIAS_ADDR`, `OUT8_ADDR`; window size; `out_shift` |
| `src/NPU.cpp` | shift register; `BIAS_ADDR` writes broadcast into a column of `C`; `OUT8_ADDR` reads compute four packed output bytes (`out8()`); `reset()` keeps the shift |
| `src/CPU.cpp` | the custom-0 switch, duplicated in `read()` and `read_one()`, moved into one `CPU::run_npu_op()` that both call; funct3 4 and 5 added there. `reg_usage()`: funct3 5 reads `rs2`. `mem_class()`: funct3 4 → `MEM_BANK_C` (writes `C`, its own NPU-side port, so it can co-issue with `load_a` / `load_b`); funct3 5 → `MEM_STORE` (same port as `store_c` and scalar stores) |
| `include/CPU.h` | `MEM_BANK_C` in `MemClass`; `run_npu_op()` |
| `scripts/lib/trace_harness.cpp` | inline NPU tags for funct3 4 and 5 |
| `tests/npu/npu_isa.h` (new) | C helpers for all six custom-0 instructions and the MMIO addresses. Same role as `tests/simt/simt_isa.h`. The existing NPU templates keep their own copies and are not changed. |
| `tests/npu/c_tests/output_stage_template.c` (new), `tests/npu/run_npu_tests.py` | the test in §6 |
| `CLAUDE.md`, NPU section | describe the two instructions (only when asked to update it) |

SM lanes still fault on all custom-0 instructions: the NPU stays a host-driven device.

## 5. How a GEMM uses it

For `Y[M×N] = X[M×K] · W[K×N] + bias`, all operands tile-major, padded to multiples of 16
(the full design of this loop is in [gemm.md](gemm.md)):

```
set DIM_K = 16 once
for each column of output tiles J:
    set DIM_N = real columns in tile column J
    for each row of output tiles I:
        set DIM_M = real rows in tile row I
        load_bias(&bias[16·J])                # C = bias (or RESET if no bias)
        for each K tile:
            load_a(X tile (I, k)); load_b(W tile (k, J)); MAC
        store_c8(&Y tile (I, J), shift)       # int8, tile-major, padding zeroed
```

`DIM_M` and `DIM_N` only change at the last tile row or column, so in the steady state
the inner loop is the same `load_a` / `load_b` / `MAC` group as `matmul_template_2.c`.

## 6. Tests (in 32I, independent of the lego library)

`python3 run_npu_tests.py --output-stage [--seed S] [--opt O2]`: 19 single-tile cases,
each checked bit-exact against an independent Python model of §1, on the scalar host and
the superscalar host at issue widths 1, 2, 4 and 10.

- `shift` = 0, 1, 7, 31;
- bias zero, positive, negative;
- saturation at both ends (all −128 / 127 inputs);
- rounding: accumulators exactly on a half before the shift, both signs;
- edge masks: (`M`, `N`) = (5, 16), (16, 3), (1, 1); padding must come out 0;
- MMIO-only path (bias port, shift register, `OUT8_ADDR` read with plain loads), full
  tile and a 7 × 9 edge;
- `RESET` after a bias preload clears it (second tile computed without bias);
- two MACs accumulate on top of the bias (`C = bias + 2·A·B`);
- `load_bias` alone, read back through `store_c` as int32: every row equals the bias.

The test was run against the code before each change and failed for the expected
reason. Deliberately broken versions (rounding removed; bias broadcast to row 0 only)
are caught.

Regression, all passing: NPU templates 1 and 2 (scalar and superscalar),
`run_npu_tests_2.py`, SIMT/grid (104 tests), basic and opt-level suites, riscv-arch-test
(47).

## 7. Next steps (not part of this change)

- **Memory hierarchy.** One instruction per tile transfer models a 256-byte-per-cycle
  path, which is realistic between the NPU and an on-chip SRAM, but not to DRAM (roughly
  16 – 64 bytes per cycle for LPDDR/DDR at 1 GHz). A later design file should model
  DRAM → SRAM traffic: SRAM size, DRAM link width, and how much each loaded tile gets
  reused.
- **Double-buffered accumulator.** Two `C` buffers, so the output stage drains one tile
  while the array accumulates the next (as in Ascend's L0C). The gain is about one cycle
  per tile while transfers cost one cycle. It becomes worthwhile once the memory
  hierarchy above makes stores take real time, so it should be measured together with
  that model.
