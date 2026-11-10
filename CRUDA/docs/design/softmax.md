# Lego: softmax (`nn_softmax`) — GPU, row-wise

Softmax along each row of a tensor, in integers:

```
m      = max_c X(r, c)                          (row maximum)
e(r,c) = table[m − X(r, c)]                     (exp by table, uint16)
S(r)   = Σ_c e(r, c)                            (row sum, uint32)
Y(r,c) = min(127, (e · 128 + S/2) / S)          (probability × 128, rounded)
```

`X` and `Y` are `rows × cols` int8 tensors in the library's padded tile layout. The
output is a probability in steps of 1/128: 64 means 0.5, 127 means "≥ 127/128". `table`
holds 256 uint16 values, `table[d] = round(2¹⁵ · exp(−d · s_in))` for an input of scale
`s_in`; like lut8, the lego doesn't know the function, the caller builds the table
offline. Each row of `Y` sums to about 128 (not exactly, because each element is rounded
on its own).

This is the library's first **row-wise** GPU lego: every output depends on its whole row,
so one thread does one row. Shared rules are in [00-overview.md](00-overview.md); the
pattern (kernel + wrapper, launch splitting, padding never touched) is lut8's
([lut8.md](lut8.md)).

## 1. Operation

For each row `0 ≤ r < rows`, over its real columns `0 ≤ c < cols`:

```
m    = max_c X(r, c)                        int8
d    = m − X(r, c)                          0 … 255
e    = table[d]                             uint16
S    = Σ_c e                                uint32, ≥ table[0] ≥ 1
Y    = (e · 128 + S / 2) / S                unsigned divide (floor), so halves round up
Y    = min(Y, 127)                          Y = 128 only when e = S (one element holds it all)
```

Padding is neither read nor written: padded columns of a row don't take part in its max
or its sum, and keep their value (zero) in `Y`. That matters downstream: a probability
matrix is usually the left operand of a matmul (`P · V`), where its padded columns must be
zero.

**Why subtract the row max.**
- `d = m − x` is always in 0 … 255, so a 256-entry table covers every input.
- `exp(x − m) ≤ 1`, so with `table[0] = 2¹⁵` every entry fits a uint16 and the row sum
  can't overflow for any allowed `cols` (§2).
- The result is unchanged: `exp(x − m) / Σ exp(x − m) = exp(x) / Σ exp(x)`.

**Example** (one row, `s_in = 1/8`, so `table[d] = round(32768 · e^(−d/8))`):

| x | 16 | 8 | 0 |
|---|---|---|---|
| d = 16 − x | 0 | 8 | 16 |
| e = table[d] | 32768 | 12055 | 4435 |
| S | 49258 | | |
| Y = (e·128 + 24629) / 49258 | **85** | **31** | **12** |
| float softmax × 128 | 85.15 | 31.33 | 11.52 |

## 2. Interface

```c
int nn_softmax(const int8_t*   X, uint32_t ldt_x,
               int8_t*         Y, uint32_t ldt_y,
               uint32_t rows, uint32_t cols,
               const uint16_t* table);
```

| Parameter | Meaning |
|---|---|
| `X`, `ldt_x` | input tensor view and its tiles per tile-row |
| `Y`, `ldt_y` | output tensor view and its tiles per tile-row |
| `rows`, `cols` | real tensor size; softmax runs along each row |
| `table` | 256 uint16 values, `table[d] = f(d)` for `d = max − x`; `table[0] ≥ 1` |

Returns 0, or -1 without launching anything if a check fails: `rows ≥ 1`;
`1 ≤ cols ≤ 65535`; `X, Y, table` non-null; `table[0] ≥ 1`; `ldt_x`, `ldt_y` each
`≥ NN_PAD16(cols)/16`; if `X == Y` then `ldt_x == ldt_y`.

- **`table[0] ≥ 1`** guarantees `S ≥ 1` (the row's maximum contributes `table[0]`), so
  the divide never sees 0. The wrapper reads this one value on the host.
- **`cols ≤ 65535`** keeps everything in uint32: `S ≤ 65535 · 65535 < 2³²` and
  `e · 128 + S/2 < 2²³ + 2³¹`. Any table works, monotone or not: `e` is always one of
  the terms of `S`, so `e ≤ S` and `Y ≤ 128` before the clamp.
- **In place is allowed** (`Y == X`, same `ldt`): a row belongs to one thread, which
  reads the row in its first two passes and, in the third, reads each element just
  before overwriting it.
- Any other overlap of `X`, `Y`, `table` is not allowed (not checked).

## 3. How it runs on the GPU

**One thread per row**, three passes over the row's columns:

```
i = simt_global_id()
if simt_split(i < count):
    r = base + i
    m = −128
    for c in 0 … cols−1:  x = X[nn_off(r, c, ldt_x)];  m = m + (x > m) · (x − m)
    S = 0
    for c in 0 … cols−1:  S += table[m − X[nn_off(r, c, ldt_x)]]
    for c in 0 … cols−1:
        e = table[m − X[nn_off(r, c, ldt_x)]]
        y = (e · 128 + S / 2) / S
        Y[nn_off(r, c, ldt_y)] = y − (y > 127)
simt_join()
```

- **The loops are allowed.** Their bound `cols` comes from the args struct, so every lane
  runs the same trip count and the loop branch is warp-uniform. The max and the clamp are
  branch-free (comparisons as 0/1), as overview §4 requires; `y ≤ 128`, so subtracting
  `(y > 127)` is the whole clamp.
- **The third pass reads `X` and the table again** instead of keeping `e` from the second
  pass: a thread has no room for a row of values (1 KB stack, no local arrays), and there
  is no shared memory yet.
- **No reductions across threads.** A row is never split between threads, so no
  barrier or shared memory is needed. That is what the hardware supports today
  (`BAR` is not implemented).

**Blocks of 32 threads** (one warp), not the overview's 128. A row-wise lego has only
`rows` threads, often a few hundred, and device time is the busiest SM's. With 32-thread
blocks the rows spread over all 4 SMs much sooner:

| rows | 128-thread blocks | busiest SM | 32-thread blocks | busiest SM |
|---|---|---|---|---|
| 144 | 2 blocks | 4 warps | 5 blocks | 2 warps |
| 512 | 4 blocks | 4 warps | 16 blocks | 4 warps |

It also keeps the per-block issue cap (below) from limiting `cols`: a block is one warp's
work. `nn.h` gets `NN_ROW_THREADS` = 32 and `NN_MAX_ROWS_PER_LAUNCH` = 65,535 × 32 =
2,097,120 (overridable with `-D`); bigger tensors take several launches that differ only
in `base` and `count`, as in lut8.

**Per-block issue cap.** `GridLauncher` abandons a block after 1,000,000 issued
instructions (a runaway-kernel guard). One block here issues about `179 · cols` (§7), so
rows up to about 5,500 columns fit. Longer rows fail with a GridLauncher timeout until
the cap is raised, which the overview already plans (§8) together with the cycle
counter. Not raised by this lego.

## 4. Additions to `nn.h`

`NN_ROW_THREADS`, `NN_MAX_ROWS_PER_LAUNCH`, `nn_row_blocks(rows)` (`ceil(rows / 32)`,
`always_inline`), and the `nn_softmax` prototype.

## 5. Python reference

```python
def nn_softmax(X, table):   # X: (rows, cols) int8; table: 256 uint16
    m = X.max(axis=1, keepdims=True)
    e = table[m - X]
    S = e.sum(axis=1, keepdims=True)
    return np.minimum((e * 128 + S // 2) // S, 127)   # all non-negative: // is the C divide

def exp_table(s_in):        # table[d] = round(2**15 * exp(-d * s_in))
```

`exp_table` uses floating point, but only to build a constant table; the bit-exact check
is against the table, not against a float softmax.

## 6. Tests

**Tables.** Most cases use a **random** table (entries 1 … 65535), for the same reason as
lut8: a wrong index then gives a wrong output instead of an almost-equal one. Special
tables where they matter:
- **exp** (`exp_table(1/8)`): the realistic case;
- **`table[0] = 1`, rest 0**: only the maxima count, so ties share 128 exactly
  (2 maxima → 64 each, 3 → 43);
- **all 65535**: the largest possible sums.

**Default:**

| Case | rows × cols | Notes |
|---|---|---|
| one element | 1 × 1 | output 127 |
| one column | 40 × 1 | every row is one element: all 127 |
| one row | 1 × 16 | |
| edges | 17 × 33 | partial tiles both ways; padded columns must not count |
| partial block | 5 × 20 | 5 of 32 lanes active |
| exactly one block | 32 × 16 | |
| many blocks | 300 × 20 | 10 blocks over 4 SMs |
| long rows | 3 × 4000 | 250 column tiles per row |
| exp table | 40 × 64 | |
| ties | 20 × 30 | `table[0] = 1` table, rows with 1, 2 and 3 maxima |
| uniform rows | 20 × 3 | all `x` equal: every output `(128 + 1) / 3 = 43` |
| full range | 16 × 16 | rows holding both −128 and 127, so `d = 255` is used |
| all negative | 20 × 30 | every row's max is below 0, so a max that doesn't start at −128 shows (uniform rows can't catch it: every `d` shifts equally) |
| largest sums | 2 × 4000 | all-65535 table |
| views | 20 × 40 | `X`, `Y` slices of wider buffers, `ldt_x ≠ ldt_y` |
| in place | 20 × 40 | `Y == X` |
| launch splitting | 45 × 10, `-DNN_MAX_ROWS_PER_LAUNCH=10` | 5 launches (10, 10, 10, 10, 5) |
| invalid | `rows = 0`; `cols = 0`; `cols = 65536`; `table = NULL`; `table[0] = 0`; `ldt_x` too small; `X == Y` with different `ldt` | return -1, `Y` untouched |

**Checks** as in lut8.md §6: `Y` starts all poison (padding included) and every
non-element byte must still be poison afterwards; `X`'s padding and the area outside its
view hold random junk, so a pass that strays into padding (into the max or the sum) gives
wrong outputs. stderr must be empty. Default cases also run on the superscalar host at
widths 1/2/4/10.

**Large (`--large`, scalar host only):** 1024 × 128; 2048 × 1000; and 2,097,200 × 1 in
place (more rows than one launch holds, so the wrapper must really split; in place
because two such tensors wouldn't fit in RAM).

## 7. Cost

- **Kernel (measured, `-O0`):** `softmax_kernel` is 231 instructions. Its three column
  loops are 47 + 45 + 87 = **179 warp instructions per column**, for 32 rows at once;
  the divide is one `divu`. The rest of the kernel runs once per row. Its only branches
  are the split guard and the three loop branches (`c < cols`, the same in every lane).
- **Estimate:** with full warps on all 4 SMs,

  ```
  device cycles ≈ ceil(rows / 32) / 4 × 179 × cols ≈ 1.4 × rows × cols
  ```

  about the same per element as eltwise. With few rows, fewer SMs are busy: a
  144 × 144 softmax is 5 warps, 2 on the busiest SM: ≈ 2 × 179 × 144 ≈ 51,600 cycles.
- **Measured** (whole-program cycles, scalar host):

  | Case | Model `warps on busiest SM × 179 × cols` | Measured |
  |---|---|---|
  | 1 × 16 vs 1 × 1 | 15 columns × 179 = 2,685 more | 3,062 − 377 = 2,685 |
  | 3 × 4000 | 1 × 179 × 4000 = 716,000 | 716,199 |
  | 1024 × 128 | 8 × 179 × 128 = 183,296 | 183,942 |
  | 2048 × 1000 | 16 × 179 × 1000 = 2,864,000 | 2,865,159 |

  The remainder is the once-per-row part of the kernel (about 80 instructions per warp)
  plus host start-up.

## 8. Correctness notes

- Each row is handled by exactly one thread and each thread writes only its own row, so
  block and warp order don't matter.
- All arithmetic is exact in uint32 (§2), so the Python reference matches bit for bit.
- The divide rounds half up: `(e·128 + S/2) / S` with `S/2` itself rounded down. The
  reference uses exactly this expression (`//` on non-negative values is C's unsigned
  divide).

## 9. Limitations and next steps

- **No mask.** Causal or padding masks (setting some scores to −∞) are not supported. A
  row's padding is already excluded; a general mask can be added as a parameter later.
- **Output scale fixed at 1/128.** That is the finest step an int8 probability in
  [0, 1] can have. A different scale would be an extra shift parameter.
- **One thread per row** leaves SMs idle when `rows` is small. Splitting a row across a
  warp needs cross-lane reductions (shared memory, `BAR`, or a warp-shuffle instruction),
  which the GPU doesn't have yet.
- **Cost at `-O0`** and the re-reads in pass 3: see lut8.md §9. Hoisting the row's
  part of `nn_off` out of the loops would also save instructions.
