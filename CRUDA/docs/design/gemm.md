# Lego: GEMM (`nn_gemm`) — NPU

General matrix multiply with bias and requantization, run on the NPU:

```
Y = X · W + bias      then rshift by `shift` and clamp to int8
```

`X` is `M × K`, `W` is `K × N`, `Y` is `M × N`, all int8 in the library's padded tile
layout. The bias is added to every row. This is an **NPU lego**: a host function that
drives the NPU tile by tile. It uses no GPU launch.

Shared rules (number format, tile layout, lego conventions, testing) are in
[00-overview.md](00-overview.md). The NPU instructions it uses, including the output
stage, are in [npu_output_stage.md](npu_output_stage.md).

## 1. Operation

For `0 ≤ m < M`, `0 ≤ n < N`:

```
acc      = bias[n]                                 (bias[n] = 0 if bias == NULL; preloaded into the NPU accumulator)
acc      = acc + Σ_{k=0}^{K-1} X(m, k) · W(k, n)   (int32, one MAC per K tile)
Y(m, n)  = sat8(rshift(acc, shift))                (NPU output stage)
```

Padding elements of `Y` (rows `≥ M` or columns `≥ N` inside the last tile row or
column) are written as 0.

## 2. Interface

```c
int nn_gemm(const int8_t*  X, uint32_t ldt_x,
            const int8_t*  W, uint32_t ldt_w,
            const int32_t* bias,
            int8_t*        Y, uint32_t ldt_y,
            uint32_t M, uint32_t K, uint32_t N,
            uint32_t shift);
```

| Parameter | Meaning |
|---|---|
| `X`, `ldt_x` | input matrix view and its tiles per tile-row |
| `W`, `ldt_w` | weight matrix view (`K × N`) and its tiles per tile-row |
| `bias` | `NN_PAD16(N)` int32 values (real values, then zeros), or `NULL` for no bias |
| `Y`, `ldt_y` | output matrix view and its tiles per tile-row |
| `M`, `K`, `N` | real matrix sizes (not padded) |
| `shift` | requantization shift, 0 – 31 |

Returns 0, or -1 without touching the NPU if any check fails:
`M, K, N ≥ 1`; `X, W, Y` non-null and 4-byte aligned; `ldt_x ≥ NN_PAD16(K)/16`;
`ldt_w ≥ NN_PAD16(N)/16`; `ldt_y ≥ NN_PAD16(N)/16`; `shift ≤ 31`.

`Y` must not overlap `X`, `W` or `bias`. This is the caller's responsibility and is not
checked.

`W` is `K × N`: tile `(k, j)` holds rows `16k…` of `W` and columns `16j…`, which is how the
NPU's B operand is laid out. Weights prepared offline are simply generated in this
order. Multiplying by a transposed matrix is not covered; see §8.

## 3. How it runs on the NPU

With `Mt = NN_PAD16(M)/16`, `Kt = NN_PAD16(K)/16`, `Nt = NN_PAD16(N)/16`, and
`tile(P, ldt, r, c) = P + (r · ldt + c) · 256`:

```
DIM_K = 16                                    # K padding is zero, so full tiles are exact
for J in 0 … Nt-1:                            # one column of output tiles
    DIM_N = min(16, N − 16·J)
    for I in 0 … Mt-1:                        # one output tile
        DIM_M = min(16, M − 16·I)
        if bias: load_bias(bias + 16·J)       # C = bias, broadcast down every row
        else:    RESET                        # C = 0
        for k in 0 … Kt-1:
            load_a(tile(X, ldt_x, I, k))
            load_b(tile(W, ldt_w, k, J))
            MAC                               # C += A · B
        store_c8(tile(Y, ldt_y, I, J), shift) # bias + shift + clamp, padding zeroed
```

- **The tile starts from its bias.** `load_bias` sets the accumulator to the bias, so
  it replaces `RESET`: one instruction per output tile either way. The bias check is on
  a host argument, so it's the same branch for every tile.
- **Loop order.** `J` outer, `I` inner; with the bias preloaded per tile the order
  doesn't affect the instruction count. It is kept so that a later operand-reuse
  optimisation (keeping a `W` tile column resident) has the natural order already.
- **Why `DIM_K` stays 16.** Both operands' padding is zero (overview §2), so the padded
  part of the last K tile adds nothing to the sum.
- **Why `DIM_M` / `DIM_N` change at the edges.** They are the output stage's
  valid-region mask. They're 16 everywhere except the last tile row and column, where
  they make the output stage write zeros into the padding.
- In the steady state the inner loop is the same `load_a` / `load_b` / `MAC` group as
  `32I/tests/npu/c_tests/matmul_template_2.c`, which the superscalar host co-issues.

## 4. Python reference

```python
def nn_gemm(X, W, bias, shift):
    """X: (M, K) int8. W: (K, N) int8. bias: (N,) int32 or None.
    Returns (M, N) int8."""
```

Computes in int64, asserts every accumulator fits in int32 (catches a test that breaks
the overview's range rule), then applies the same `rshift` and `sat8`. It works on plain
row-major arrays; the driver converts to and from the tile layout.

## 5. Tests

**Default:**

| Case | M × K × N | Notes |
|---|---|---|
| one element | 1 × 1 × 1 | one tile, 1 × 1 valid region |
| one full tile | 16 × 16 × 16 | no edges |
| edges everywhere | 17 × 33 × 5 | partial tiles in M, K and N |
| multi-tile exact | 48 × 64 × 32 | 3 × 4 × 2 tiles, no edges |
| K = 1 | 20 × 1 × 20 | one K tile, mostly padding |
| long K | 8 × 300 × 24 | 19 K tiles |
| bias | with and without | `NULL`, and random int32 bias |
| edge padding | 5 × 16 × 3 with non-zero bias | padding must still be zero |
| views | `X` a column slice of a wider buffer; `Y` written into a column slice of a wider buffer | `ldt > needed`; the wider `Y` buffer's other tiles are poisoned and must be unchanged |
| saturation | all `X = W = -128`; `X = -128`, `W = 127`; `shift = 0` | outputs pinned at +127 and −128 |
| rounding | accumulators exactly on a half before the shift, both signs | half-toward-+∞ rule |
| shift range | `shift` = 0 and 31 | |

For random-data cases the driver picks `shift` so that most outputs land inside
[-128, 127] rather than saturating (`shift ≈ log2(√K · 128)`), so the check actually
compares values instead of a matrix of ±127.

Every default case also runs on the superscalar host at issue widths 1, 2, 4 and 10.

**Large (`--large`):** 256 × 512 × 512, 128 × 1536 × 384, 1024 × 64 × 1024.

## 6. Cost

- Multiply-accumulates: `M · K · N` (done by the NPU, 4,096 per `MAC`).
- NPU instructions: `Mt · Nt · Kt` × (`load_a` + `load_b` + `MAC`), plus `Mt · Nt` ×
  (`load_bias` or `RESET` + `store_c8` + `DIM_M` write), plus `Nt` × `DIM_N` write.
- Host cycles ≈ `Mt · Nt · (Kt · c_k + c_0) + Nt · c_J`, where `c_k` is the instructions
  in one K-loop iteration (NPU instructions + address arithmetic + loop branch), `c_0` the
  per-output-tile overhead, and `c_J` the per-column overhead.
- **Measured** (`-O2`, fitted to the three large test shapes; `c_J` is negligible):

  | Host | `c_k` (per K step) | `c_0` (per output tile) |
  |---|---|---|
  | scalar | ≈ 7.0 | ≈ 15 |
  | superscalar, width 4 | ≈ 2.0 | ≈ 4 |
  | superscalar, width 10 | ≈ 1.0 | ≈ 4 |

  The compiled K loop is 7 instructions: `load_a`, `load_b`, the `MAC` trigger store,
  three address/counter updates and the branch. At width 10 one K step issues per cycle.
  Example: 256 × 512 × 512 (16 · 32 · 32 = 16,384 K steps, 512 output tiles) runs in
  122,957 cycles scalar and 18,631 at width 10. These are whole-program cycles (start-up
  included, a few dozen) until `rdcycle` exists.
- Tile transfers cost one instruction each. A bus model would add the cost of moving
  256 bytes per `load_a` / `load_b` / `store_c8`; that is a separate, later decision
  (see [npu_output_stage.md](npu_output_stage.md) §2).

## 7. Correctness notes

- **Padding must already be zero** in `X`'s columns `≥ K` and `W`'s rows `≥ K`. This is
  guaranteed by the overview's padding rule. The lego does not check it, because it
  would cost a full pass over both inputs.
- The NPU holds one A tile, one B tile and one accumulator. Every output tile starts
  from `load_bias` or `RESET`, and this lego sets `DIM_*` before use, so calls don't
  depend on what ran before.

## 8. Limitations and next steps

- **Transposed operands.** `W` must be stored `K × N`. Multiplying by the transpose of a
  matrix that was produced at run time (e.g. `A · Bᵀ` where `B` is another lego's
  output) needs either a transpose lego on the GPU or a transposed-load NPU instruction.
  To be decided when the first lego that needs it is designed.
- **No operand reuse.** Every `MAC` reloads both tiles: `X`'s tile row is reloaded for
  every output column, and `W`'s tile column for every output row. Operand buffers
  (keeping a tile across several `MAC`s) would be an NPU hardware change.
- **Bus cost** of tile transfers is not modelled yet (see §6).
