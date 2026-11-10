# Sparse-ViT-Det: one ViT encoder block

This file designs **one transformer encoder block** of Sparse-ViT-Det, built only from
existing legos. It is the first network program in CRUDA. The legos stay generic; every
model-specific shape, buffer and scale lives here.

The goal is accelerator design, so the program has two jobs:

1. **Correctness:** the block's output must match a Python reference bit for bit.
2. **Cost:** every lego call is timed with `nn_cycles()` ([cycle_counter.md](../../docs/design/cycle_counter.md)),
   so one run prints where the cycles go. The full model's encoder is estimated as
   12 × one block.

The guidance CNN, the token sampler and the gather that produce the 144 tokens are not
part of this file. They need new legos (convolution, top-k, gather) and come later.

## 1. The block

ViT-Small dimensions, as in the model summary:

| Symbol | Meaning | Value |
|---|---|---|
| `T` | tokens | 144 |
| `D` | embedding width | 384 |
| `H` | attention heads | 6 |
| `d` | head width, `D / H` | 64 |
| `F` | MLP hidden width | 1536 |

It is a standard pre-norm block:

```
X1 = X  + Wo · Attention(LN1(X))
Y  = X1 + W2 · GELU(W1 · LN2(X1))
```

**Assumptions to confirm:** every block uses **global** attention over all `T` tokens:
no windowed attention, no relative position bias, no dropout (inference). Positional
embeddings are added before the first block, outside this file. If some blocks of the
real model use windowed attention, that changes the attention steps of §2 only (more,
smaller score matrices) and can be added as a second variant.

## 2. Call sequence

22 lego calls. `QKV` holds Q, K and V side by side, so each head's Q_h, K_h and V_h
are column slices, with no copies (§3).

| # | Step | Lego | Shape (M × K × N, or rows × cols) | In → out |
|---|---|---|---|---|
| 1 | LN₁ | `nn_layernorm` | 144 × 384 | `X` → `XN` |
| 2 | QKV projection | `nn_gemm` | 144 × 384 × 1152 | `XN` → `QKV` |
| 3–8 | scores `S_h = Q_h · K_hᵀ`, h = 0…5 | `nn_gemm_bt` | 144 × 64 × 144 | `QKV` → `S` (row block h) |
| 9 | softmax, **all heads in one call** | `nn_softmax` | 864 × 144 | `S` → `S` (in place, becomes `P`) |
| 10–15 | `O_h = P_h · V_h`, h = 0…5 | `nn_gemm` | 144 × 144 × 64 | `S`, `QKV` → `O` (column slice h) |
| 16 | output projection | `nn_gemm` | 144 × 384 × 384 | `O` → `XN` |
| 17 | residual | `nn_add` | 144 × 384 | `X`, `XN` → `X` (in place) |
| 18 | LN₂ | `nn_layernorm` | 144 × 384 | `X` → `XN` |
| 19 | FC1 | `nn_gemm` | 144 × 384 × 1536 | `XN` → `HID` |
| 20 | GELU | `nn_lut8` | 144 × 1536 | `HID` → `HID` (in place) |
| 21 | FC2 | `nn_gemm` | 144 × 1536 × 384 | `HID` → `XN` |
| 22 | residual | `nn_add` | 144 × 384 | `X`, `XN` → `X` (in place) |

The block's output is `X` itself, so 12 blocks
chain without copies.

- **Softmax as one call.** The six 144 × 144 score matrices are stored on top of each
  other as one 864 × 144 tensor. Head h's `S_h` is the row slice starting at tile row
  `9h`. One call over 864 rows makes 27 one-warp blocks, so all 4 SMs stay busy. Six
  separate calls would run 5 blocks each, with 2 on the busiest SM, and cost ≈ 310k
  cycles instead of ≈ 181k (§8).
- This needs **`T` to be a multiple of 16.** Otherwise the padding rows of one head
  would sit between the real rows of the next, and the call would treat them as real
  rows. The model's 144 is a multiple of 16. The program checks it and returns an error
  otherwise.
- **The 1/√d scale** of attention is a power of two (1/8), so it is just 3 more bits on
  the score GEMM's shift (§5). This needs `d` to be 16, 64 or 256 (a multiple of 16 for
  the column slices, and a power of 4 so that √d is a power of two).

## 3. Buffers

All tensors are tile-major (overview §2). `ldt` is in tiles; every size below is a
multiple of 16, so there is no padding at the real dimensions.

| Buffer | Shape | `ldt` | Bytes | Holds |
|---|---|---|---|---|
| `X` | 144 × 384 | 24 | 55,296 | block input, then `X1`, then the output |
| `XN` | 144 × 384 | 24 | 55,296 | LN₁ out, then out-proj out, LN₂ out, FC2 out |
| `QKV` | 144 × 1152 | 72 | 165,888 | Q (tile cols 0–23), K (24–47), V (48–71) |
| `S` | 864 × 144 | 9 | 124,416 | scores, then probabilities (in place) |
| `O` | 144 × 384 | 24 | 55,296 | the six heads' outputs side by side |
| `HID` | 144 × 1536 | 96 | 221,184 | FC1 out, then GELU out (in place) |

Views used by the per-head calls, each with the parent buffer's `ldt`:

```
Q_h = QKV + (4h)      · 256      K_h = QKV + (24 + 4h) · 256      V_h = QKV + (48 + 4h) · 256
S_h = S   + (9h · 9)  · 256      O_h = O   + (4h)      · 256
```

**Totals:** activations ≈ 677 KB. Weights for one block: `Wqkv` 384 × 1152, `Wo`
384 × 384, `W1` 384 × 1536, `W2` 1536 × 384, all int8: 1.77 MB, plus 14 KB of int32
biases and the LN parameters. All 12 blocks' weights would be ≈ 21 MB, which still fits
in the 64 MB of RAM. The test in §7 stores one block.

Every in-place step is one the lego allows (`Y == X` with the same `ldt`: softmax,
lut8, add).

## 4. Interface

```c
struct vit_dims {
    uint32_t T, D, heads, d_head, hidden;      // D == heads * d_head
};

struct vit_weights {
    const int8_t*  ln1_gamma;  const int32_t* ln1_beta;  uint32_t ln1_shift;
    const int8_t*  w_qkv;      const int32_t* b_qkv;     uint32_t qkv_shift;
    uint32_t       score_shift;
    const uint16_t* exp_table;                            // softmax table
    uint32_t       pv_shift;
    const int8_t*  w_o;        const int32_t* b_o;       uint32_t o_shift;
    uint32_t       res1_sa, res1_sb, res1_shift;
    const int8_t*  ln2_gamma;  const int32_t* ln2_beta;  uint32_t ln2_shift;
    const int8_t*  w_fc1;      const int32_t* b_fc1;     uint32_t fc1_shift;
    const int8_t*  gelu_table;                            // lut8 table
    const int8_t*  w_fc2;      const int32_t* b_fc2;     uint32_t fc2_shift;
    uint32_t       res2_sa, res2_sb, res2_shift;
};

struct vit_bufs { int8_t *x, *xn, *qkv, *s, *o, *hid; };

// Runs one block in place on bufs->x. cycles[i] = nn_cycles() around call i
// (VIT_CALLS = 22 entries). Returns 0, -1 if the dims are invalid (D != heads *
// d_head, d_head or T not a multiple of 16), or i + 1 if lego call i returned -1
// (nothing after it runs).
int vit_block(const struct vit_dims* dims, const struct vit_weights* w,
              const struct vit_bufs* b, uint32_t* cycles);
```

- **Dimensions are run-time values**, so the same code runs the small test block and
  the real one (§7).
- **Shifts and tables belong to the weights**, so each of the 12 blocks can have its
  own.
- **Files:** `models/sparse_vit_det/vit_block.c` (host code, `-O2`, calls the legos),
  `vit_block.h`, `ref/vit_block_ref.py` (§6), `tests/t_vit_block.c` and
  `run_vit_block.py` (§7). It links the existing lego files unchanged: `gemm.c`
  (`-O2`), and `layernorm.c`, `softmax.c`, `lut8.c`, `eltwise.c` (`-O0`).

## 5. Quantization plan

Every tensor is int8 with a **power-of-two scale**: the value `q` stands for
`q · 2^−f`, where `f` is the tensor's number of fractional bits. A GEMM's int32
accumulator has `f_x + f_w`, and a shift `s` gives an output with `f_x + f_w − s`. So
each shift follows from the `f` the output should have, and the whole plan is a table
of `f` values.

**Weights are synthetic** for now (no trained checkpoint): int8 drawn from a normal
distribution with standard deviation 32, clipped to ±127. They stand for real weights
with `f_w = round(5 + log2(√K))`, i.e. a real standard deviation of about `1/√K`
(the usual initialization). With that choice a GEMM keeps the size of its input, so no
step drifts towards saturation or towards zero. Biases are small random int32 values
at the accumulator's scale. Trained weights would change only the `f_w` column and
the shifts that follow from it.

| Tensor | `f` | Range | How it is produced |
|---|---|---|---|
| `X` (block input and output) | 4 | ±8 | test input: normal, std 32 (real std 2) |
| LN₁ / LN₂ out | 5 | ±4 | `gamma` = 32 · γ (γ random in [0.5, 1.5]), `beta` small, `shift = 0` |
| `Wqkv`, `Wo`, `W1` (K = 384) | 9 | | synthetic |
| `W2` (K = 1536) | 10 | | synthetic |
| Q, K, V | 5 | ±4 | `qkv_shift = 5 + 9 − 5 = 9` |
| scores `S` | 4 | ±8 | accumulator has `f = 10`, times 1/8: `score_shift = 10 + 3 − 4 = 9` |
| probabilities `P` | 7 | [0, 1) | fixed by softmax (steps of 1/128) |
| head outputs `O` | 5 | ±4 | `pv_shift = 7 + 5 − 5 = 7` |
| out-proj out | 5 | ±4 | `o_shift = 5 + 9 − 5 = 9` |
| `X1 = X + out` | 4 | ±8 | `nn_add(X, sa = 1, out, sb = 0, shift = 1)`: both at `f = 5`, then back to 4 |
| FC1 out | 5 | ±4 | `fc1_shift = 5 + 9 − 5 = 9` |
| GELU out | 5 | ±4 | table below |
| FC2 out | 5 | ±4 | `fc2_shift = 5 + 10 − 5 = 10` |
| `Y = X1 + FC2 out` | 4 | ±8 | as `X1` |

**Tables** (built in Python, passed in as data):

- **Softmax:** scores have scale `s_in = 2^−4`, so the table is softmax.md's
  `exp_table(1/16)`: `table[d] = round(2¹⁵ · e^(−d/16))`.
- **GELU:** input and output at `f = 5`:
  `table[i] = sat8(round(GELU((i − 128) / 32) · 32))` (lut8.md §2).

**Sanity check, reported but not pass/fail:** the Python side also runs the same block
in floating point with the real-valued weights (`q · 2^−f`). It prints, for every step,
the fraction of saturated outputs and the correlation between the int8 and float
results. This shows whether the plan is reasonable. It is not a test: the bit-exact
comparison against the integer reference is.

## 6. Python reference (`ref/vit_block_ref.py`)

```python
def vit_block(X, w, dims):
    """X: (T, D) int8. w: the weights, shifts and tables of section 5.
    Returns (T, D) int8 and every intermediate tensor (for debugging)."""
```

It chains the existing `nn_ref` functions (`nn_layernorm`, `nn_gemm`, `nn_gemm_bt`,
`nn_softmax`, `nn_lut8`, `nn_add`) in the order of §2, with the same integer semantics.
The per-head slices are plain NumPy column slices; the reference never needs the tile
layout. It also holds the weight generator, the shift table of §5, both tables, and the
float version of §5's sanity check.

## 7. Tests (`run_vit_block.py`)

Like the lego tests (overview §7): generate the input and weights in Python, write them
to `data.bin`, build, run on `32I/build/run_test`, dump `X` and the cycle array, and
compare. Checks: `X` matches the reference bit for bit, padding is zero, a poisoned
guard region after `X` is untouched, `a0 = 0`, stderr is empty.

| Case | T, D, heads × d, F | Runs on |
|---|---|---|
| small | 48, 64, 4 × 16, 120 | scalar, superscalar widths 1/2/4/10 |
| real (`--real`) | 144, 384, 6 × 64, 1536 | scalar and width 10; prints the cycle table |
| invalid | `T = 40`; `D ≠ heads · d` | returns -1, `X` untouched |

- The small case has `F = 120`, so FC1's output, GELU and FC2's K dimension have real
  padding (columns 120–127). That exercises the zero-padding rule between legos.
  `T` and `d` must be multiples of 16 (§2), so they have none.
- **Debugging a mismatch:** the block reuses buffers, so intermediates are overwritten.
  `t_vit_block.c` built with `-DVIT_STOP_AFTER=k` stops after call `k`, and the driver
  then compares that call's output buffer with the reference's intermediate. Every lego
  is already tested on its own, so a mismatch here most likely comes from a view, an
  `ldt` or a shift, and this finds the step.

**Output of `--real`:** one line per call (step, lego, unit, cycles, % of the block),
then the totals for the NPU (GEMM) calls and the GPU calls, and the 12-block
extrapolation.

## 8. Expected cost

Estimates from the measured cost models of each lego (gemm.md §6, layernorm.md,
softmax.md, lut8.md, eltwise.md). The run will replace them with measurements.

| Part | Scalar host | Issue width 10 |
|---|---|---|
| GEMMs: QKV 119k, scores 21k, P·V 17k, out-proj 40k, FC1 158k, FC2 148k | ≈ 502k | ≈ 77k |
| LayerNorm × 2: 5 one-warp blocks each, 2 on the busiest SM: 2 × (189 · 384 + 1,590) | ≈ 297k | ≈ 297k |
| softmax, 864 rows: 27 blocks, 7 on the busiest SM: 7 × 179 · 144 | ≈ 181k | ≈ 181k |
| GELU, 221,184 elements × 0.875 | ≈ 194k | ≈ 194k |
| residual adds × 2, 55,296 elements each | ≈ 164k | ≈ 164k |
| **One block** | **≈ 1.34M** | **≈ 0.91M** |
| **12 blocks** | **≈ 16M** | **≈ 11M** |

The GPU parts don't depend on the host's issue width: the host is stalled during a
LAUNCH. So at width 10 the **GPU passes are ≈ 90% of the block**, and the matmuls,
which hold practically all of the block's ≈ 271M multiply-accumulates, are under 10%. Within the GPU part,
LayerNorm is the worst: 144 rows give only 5 warps, one thread per row, so most of the
GPU idles while 384-column rows are walked serially.

That is the result this block is built to measure. The obvious next steps it points to
are **fusing** element-wise work into the NPU output stage (GELU as a LUT after
`store_c8`, the residual add as a second operand), and **row-parallel** LayerNorm and
softmax (a warp per row, which needs a cross-lane reduction the GPU doesn't have yet).

## 9. Open decisions

1. **Global attention in every block** (§1): confirm against the real model.
2. **Synthetic weights** (§5): enough for correctness and cycle counts. Real trained
   weights only matter for accuracy against the float model, which is a separate study.
3. **Where the model's files live:** `CRUDA/models/sparse_vit_det/` (this file, the
   block code, its reference and driver), next to the generic `kernels/` and `ref/`.
