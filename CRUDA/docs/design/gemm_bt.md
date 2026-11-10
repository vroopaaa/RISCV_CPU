# Lego: GEMM with a transposed weight (`nn_gemm_bt`) — NPU, plus `load_b_t`

```
Y = sat8(rshift(X · Wtᵀ + bias, shift))
```

The same operation as [gemm.md](gemm.md), except the second operand is given
**transposed**: `Wt` is stored `N × K`, and the product uses its transpose (`K × N`). This
is what `A · Bᵀ` needs when `B` is produced at run time, for example a score matrix
`S = Q · Kᵀ` where `Q` and `K` are both `T × d` outputs of earlier matmuls (each row is
one item, each column one feature). `nn_gemm` cannot do it: its `W` must be stored `K × N`,
and `K` was produced `T × d`, not `d × T`.

This file covers a hardware change and the lego that uses it, because neither makes sense
without the other: a new NPU instruction, **`load_b_t`** (custom-0 funct3 6), which loads a
tile into `B` transposed, and the host function `nn_gemm_bt` built on it. Shared rules are
in [00-overview.md](00-overview.md); the NPU instructions it builds on are in
[npu_output_stage.md](npu_output_stage.md).

## 1. The decision: transpose in the NPU's load, not in a separate pass

gemm.md §8 left two options open.

**Option A, a GPU transpose lego** (`Y(c, r) = X(r, c)`, one thread per element, like
lut8), then a plain `nn_gemm` on the result.
- No hardware change, and it's generic.
- But it is a full extra pass over the tensor, plus a second buffer the size of the
  operand. Estimate for `S = Q · Kᵀ` with `Q`, `K` of `128 × 64`: the transpose touches
  8,192 elements, about 64 blocks × 4 warps × ~110 instructions / 4 SMs ≈ **7,000 device
  cycles**. The matmul itself is 8 × 8 output tiles × 4 K steps: about **2,750 cycles on
  the scalar host and ~500 at issue width 10** (gemm.md §6). The transpose would cost 2.5×
  to 14× the matmul it serves.

**Option B, a transposed tile load on the NPU** (chosen).
- In the tile layout, transposing a matrix means transposing each 16 × 16 tile and
  swapping tile coordinates: tile `(k, J)` of `Wtᵀ` is tile `(J, k)` of `Wt`, transposed.
  The tile-coordinate swap is just a different address in the host loop. The in-tile
  transpose is done by the load itself.
- **Cost: zero extra instructions.** `load_b_t` replaces `load_b` one for one; the K loop
  keeps the same 7 instructions. No extra buffer, no extra pass.
- **Hardware cost: small.** `B` is a 16 × 16 byte buffer. Writing it transposed means
  byte `(i, j)` of the incoming tile goes to position `(j, i)`: a fixed wiring permutation,
  selected by one 2:1 multiplexer per byte (256 byte-wide muxes, 2,048 mux bits). Real
  matrix engines do the same thing; transposing on load into the cube/systolic operand
  buffer is a standard feature.

Only `B` gets a transposed load. The other operand of a product like `P · V` is already
in the right orientation, and no current lego needs `Aᵀ`.

## 2. The instruction: `load_b_t` (custom-0 funct3 6)

```
.insn r 0x0B, 6, 0, x0, rs1, x0          npu_load_b_t(const int8_t* src)    (npu_isa.h)
```

Copies the 256-byte int8 tile at `rs1` into `B`, transposed:

```
B[i][j] = src[j · 16 + i]        for 0 ≤ i, j < 16
```

- Same operand rules as `load_b`: `rs1` = tile address (4-byte aligned, contiguous
  256 bytes), `rs2` unused, no register written.
- **Superscalar host:** memory class `MEM_BANK_B`, the same as `load_b`. Both write `B`,
  so at most one of them issues per window (the same structural rule `load_a` /
  `load_b` already follow).
- Nothing else changes: `DIM_K/M/N`, `MAC`, `load_bias`, `store_c8` are untouched. `B` is
  `B` however it was loaded.

**Emulator changes** (`32I/`): `CPU::run_npu_op` gets case 6; `CPU::mem_class` maps
funct3 6 to `MEM_BANK_B`; `npu_isa.h` gets `npu_load_b_t`. `reg_usage` already treats rs2
as unused for every funct3 except 2, 3 and 5. `scripts/lib/trace_harness.cpp` gets the
funct3 6 tag for `assemblyinstruction -trace`.

## 3. The lego: `nn_gemm_bt`

```c
int nn_gemm_bt(const int8_t*  X,  uint32_t ldt_x,
               const int8_t*  Wt, uint32_t ldt_wt,
               const int32_t* bias,
               int8_t*        Y,  uint32_t ldt_y,
               uint32_t M, uint32_t K, uint32_t N,
               uint32_t shift);
```

Identical to `nn_gemm` (gemm.md §2) except for the second operand:

| Parameter | Meaning |
|---|---|
| `Wt`, `ldt_wt` | weight view stored **`N × K`** (the transpose of the `K × N` matrix used), and its tiles per tile-row |

Checks are gemm's, with `ldt_wt ≥ NN_PAD16(K)/16` (a row of `Wt` has `K` elements)
instead of `ldt_w ≥ NN_PAD16(N)/16`.

**How it runs:** gemm.md §3, with the K loop changed:

```
for k in 0 … Kt-1:
    load_a(tile(X, ldt_x, I, k))
    load_b_t(tile(Wt, ldt_wt, J, k))       # tile (J, k) of Wt = tile (k, J) of Wtᵀ, transposed
    MAC
```

So along K the host walks **along** a tile-row of `Wt` (`+256` bytes per step), where
plain gemm walks down a tile-column of `W` (`+ldt_w · 256`).

**Column slices work for `Wt` too.** If `K` holds several heads side by side
(`T × (h · d)`), head `h`'s `K_h` is the view `(K + (h·d/16) · 256, same ldt)`, and
`nn_gemm_bt(Q_h, …, K_h, …)` gives `Q_h · K_hᵀ` with no copy, as long as `d` is a multiple
of 16.

**Padding.** `Wtᵀ`'s rows `≥ K` are `Wt`'s columns `≥ K`, which are padding and zero by
the overview's rule; its columns `≥ N` are `Wt`'s rows `≥ N`, also zero. So the
padded part of every tile adds nothing, as in gemm.

**Implementation.** `nn_gemm` and `nn_gemm_bt` share one host function in `gemm.c`,
`gemm_core`, with a `transposed` flag that picks one of two K loops. It is
`always_inline`, so each wrapper gets its own copy with the flag a constant and the
unused loop compiled away. (A first version that tested the flag at run time made
`nn_gemm` 20–30 cycles slower per call; inlining brought it back to its old numbers.) Both
compiled K loops are 7 instructions: `load_a`, `load_b` or `load_b_t`, the `MAC` store,
three counter/pointer updates and the branch.

## 4. Python reference

```python
def nn_gemm_bt(X, Wt, bias, shift):     # = nn_gemm(X, Wt.T, bias, shift)
```

## 5. Tests

**NPU instruction** (`32I/tests/npu/run_npu_tests.py --output-stage`, new mode 5 in
`c_tests/output_stage_template.c`): `load_bias`, `load_a`, **`load_b_t`**, `MAC`,
`store_c8`, checked against the Python model with `B` transposed. Cases: random tiles
(non-symmetric, so a missing transpose shows), an edge mask, and a `load_b` right after a
`load_b_t` on the same tile slot (the second load must fully replace `B`). Scalar and
superscalar widths 1/2/4/10, like the existing output-stage cases.

**Lego** (`run_nn_tests.py --lego gemm_bt`): every gemm case (gemm.md §5) run through
`nn_gemm_bt` with `Wt = Wᵀ` stored `N × K`, plus:

| Case | Notes |
|---|---|
| `Wt` view | `Wt` a column slice of a wider buffer with junk around it (the per-head case) |
| invalid: `ldt_wt` too small | `ldt_wt = NN_PAD16(K)/16 − 1` |

Same checks as gemm: bit-exact against the reference, zero padding, poisoned neighbours and
guard unchanged, scalar and superscalar widths 1/2/4/10.

## 6. Cost

- **Same as `nn_gemm`** (gemm.md §6): the same NPU instruction count, the same 7-instruction
  K loop. Measured on every test shape: identical cycles on the scalar host, and exactly
  1 cycle more on the superscalar host (a constant from the call, not per K step; e.g.
  "long K", 8 × 300 × 24: 436 / 227 / 136 / 90 at widths 1 / 2 / 4 / 10 against
  gemm's 436 / 226 / 135 / 89).
- Compared with option A it saves the whole transpose pass (§1) and its buffer.
- **The memory class is not observable here.** `load_b_t` is `MEM_BANK_B` because it
  writes `B`, like `load_b`. But in the GEMM K loop the next `load_a` already conflicts with
  the current one (bank A), so a window never holds two B loads, and the superscalar host
  executes a window's slots in program order, so the class only affects timing. Mapping
  funct3 6 to `MEM_NONE` instead changes neither results nor cycles in any current test;
  no test covers it.

## 7. Limitations and next steps

- **No `Aᵀ`.** A transposed `A` load would be the same change on the other operand; add
  it when a lego needs `Aᵀ · B`.
- **Bus cost** of tile transfers is still not modelled (gemm.md §8); a transposed load
  moves the same 256 bytes as a plain one.
