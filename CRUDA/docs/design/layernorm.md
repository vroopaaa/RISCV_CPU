# Lego: layer normalization (`nn_layernorm`) — GPU, row-wise

Normalizes each row of a tensor to zero mean and unit variance, then scales and shifts it
per column, in integers:

```
z(r, c) = (X(r, c) − mean_r) / std_r                 (computed exactly, see §1)
Y(r, c) = sat8(rshift(round(z · gamma[c]) + beta[c], shift))
```

`X` and `Y` are `rows × cols` int8 tensors in the library's padded tile layout,
normalized along each row. `gamma` is `cols` int8 values and `beta` is `cols` int32
values, plain arrays. `shift` requantizes to int8. The mean and the standard deviation are
the row's own (population variance, as in standard layer normalization).

This is the second **row-wise** GPU lego and follows [softmax.md](softmax.md): one thread
per row, one-warp blocks, launch splitting by rows, padding never touched. Shared rules
are in [00-overview.md](00-overview.md).

## 1. Operation

For each row, with `n = cols` and sums over its real columns only:

```
s1 = Σ x                         int32       (sum)
s2 = Σ x²                        int32       (sum of squares)
V  = n · s2 − s1²                uint64      (= n² · variance, exact, ≥ 0)
q  = isqrt(V)                    uint32      (floor of the square root; = n · std)
q  = q + (q == 0)                            (a constant row has V = 0; see "epsilon")
for each column c:
    D = n · x − s1               int32       (= n · (x − mean), exact)
    N = D · gamma[c]             int32
    t = round(N / q)             int32       (round half away from zero)
    Y = sat8(rshift(t + beta[c], shift))
```

**Why it is computed this way.** `D / q = n·(x − mean) / (n·std) = (x − mean) / std`
exactly, so `t = z · gamma` to within one rounding step. The mean is never rounded to an
integer: `n · x − s1` keeps it exact. The only approximations are `isqrt` (floor) and the
final rounding of `t`.

**Example** (one row `x = [10, 20, 30, 40]`, `gamma = 64` everywhere, `beta = 0`,
`shift = 0`):

| | |
|---|---|
| `s1 = 100`, `s2 = 3000`, `n = 4` | |
| `V = 4·3000 − 100² = 2000` | `q = isqrt(2000) = 44` (√2000 = 44.72) |
| `D = 4x − 100 = [−60, −20, 20, 60]` | |
| `N = 64·D = [−3840, −1280, 1280, 3840]` | |
| `t = round(N / 44) = [−87, −29, 29, 87]` | float `z · 64` = [−85.9, −28.6, 28.6, 85.9] |

The difference (87 vs 85.9) comes from the floor in `isqrt` (44 instead of 44.72). Its
relative size is at most `1/q`; `q` grows with the row's spread and length (here a
4-element row with a small range). On random int8 rows, `t` (clamped to int8) is never
more than 1 step from `round(z · gamma)` computed in floating point: on average 0.02 steps
at 16 columns, and under 0.001 at 384 and 1536 columns.

**Scales.** `z` has no unit. If the real gamma is `gamma · s_g`, then
`round(z · gamma) · s_g` is the real `z · γ`, so `beta` must be given in the same units
(`β / s_g`), and `Y` comes out at scale `s_g · 2^shift`. Choosing these is the caller's job.

**Epsilon.** Floating-point layer normalization divides by `√(var + ε)` (ε ≈ 10⁻⁶) to
avoid dividing by zero. Here the only case that needs it is a constant row (`V = 0`,
every `D = 0`): `q` becomes 1, every `t` is 0 and the output is `beta`, which is what
`ε` gives in floating point. For any other row, `ε` is far below one quantization step
squared, and dropping it changes nothing measurable.

**Padding** is neither read nor written: padded columns don't take part in `s1` or `s2`
and keep their value (zero) in `Y`.

## 2. Interface

```c
int nn_layernorm(const int8_t*  X, uint32_t ldt_x,
                 int8_t*        Y, uint32_t ldt_y,
                 uint32_t rows, uint32_t cols,
                 const int8_t*  gamma,
                 const int32_t* beta,
                 uint32_t shift);
```

| Parameter | Meaning |
|---|---|
| `X`, `ldt_x` | input tensor view and its tiles per tile-row |
| `Y`, `ldt_y` | output tensor view and its tiles per tile-row |
| `rows`, `cols` | real tensor size; normalization runs along each row |
| `gamma` | `cols` int8 scale values (plain array) |
| `beta` | `cols` int32 offsets in units of `gamma`'s scale, `|beta| ≤ 2²⁹` |
| `shift` | requantization shift, 0 – 31 |

Returns 0, or -1 without launching anything if a check fails: `rows ≥ 1`;
`1 ≤ cols ≤ 32767`; `X, Y, gamma, beta` non-null; `ldt_x`, `ldt_y` each
`≥ NN_PAD16(cols)/16`; `shift ≤ 31`; if `X == Y` then `ldt_x == ldt_y`.

**Ranges** (why `cols ≤ 32767` and `|beta| ≤ 2²⁹`):

| Value | Bound | Fits |
|---|---|---|
| `s2` | `n · 2¹⁴ < 2²⁹` | int32 |
| `V` | `n · s2 < n² · 2¹⁴ < 2⁴⁴` | uint64 |
| `q` | `< 2²²` | uint32, so `isqrt` needs 22 steps |
| `D` | `|n·x − s1| ≤ n · 255 < 2²³` | int32 |
| `N` | `< 2²³ · 128 = 2³⁰` | int32; `2·|N| + q < 2³²` for the rounding |
| `t + beta + 2³⁰` (inside `rshift`) | `|t| ≤ √n · 128 < 2¹⁵`, so `< 2³¹` | int32 |

`beta`'s range is the caller's responsibility (not checked: it would cost a pass over
`beta` on the host).

**In place is allowed** (`Y == X`, same `ldt`): the row's thread reads the whole row in
pass 1, and in pass 2 reads each element just before overwriting it.

## 3. How it runs on the GPU

**One thread per row**, as in softmax (one-warp blocks of `NN_ROW_THREADS` = 32,
`nn_row_blocks`, launches of at most `NN_MAX_ROWS_PER_LAUNCH` rows):

```
pass 1 (cols steps):   s1 += x;  s2 += x · x
once per row:          V = n·s2 − s1²  (64-bit);  q = isqrt(V)  (22 steps)
pass 2 (cols steps):   D, N, t, Y as in §1, store
```

- **64-bit without library calls.** `V` needs 44 bits. On RV32 gcc multiplies 64-bit
  values inline (`mul` + `mulhu`) and adds/subtracts them inline, so `n · s2 − s1²` is
  plain code. A 64-bit *divide* would call `__udivdi3`, which the bare-metal build doesn't
  have; nothing here divides 64-bit values.
- **`isqrt` is the bit-by-bit method**, a loop of exactly 22 steps (the bound is a
  constant, so the loop is warp-uniform):

  ```
  q = 0
  for b = 21 down to 0:
      cand = q | (1 << b)
      le   = (cand² ≤ V)        # 64-bit compare written as 32-bit compares
      q    = q | (le << b)
  ```

  `cand² ≤ V` is written out as `(hi < Vhi) | ((hi == Vhi) & (lo ≤ Vlo))` on the two
  32-bit halves, because gcc compiles a 64-bit `<=` into branches, and a branch on lane
  data is not allowed (overview §4). In hardware this is the standard restoring
  square-root circuit: one compare and one OR per result bit.
- **Rounded divide, branch-free:** with `sg = (N > 0) − (N < 0)`,
  `t = sg · ((2·|N| + q) / (2q))`. For `N > 0` this is `floor(N/q + 1/2)`; for `N < 0` the
  mirror image, so halves round away from zero and the result is symmetric in the sign
  of `x − mean`. Unsigned 32-bit divide, one `divu`.
- **Clamp and requantization** reuse `nn_rshift` and `nn_sat8` from eltwise.

## 4. Additions to `nn.h`

The `nn_layernorm` prototype, and the square root as a shared helper,
`nn_isqrt64(vhi, vlo)` (`always_inline`, input `< 2⁴⁴`, `NN_ISQRT64_BITS` = 22), so it can
be tested on its own (§6). Row-wise launch helpers already exist from softmax.

**The `q == 0` guard** (`q = q + (q == 0)`) is required even though no output test can
see it: dividing by zero is undefined in C. On RISC-V `divu` by zero returns all ones
without trapping, and `sg = 0` for a constant row turns that into `t = 0` anyway, so the
result would happen to be right here; other processors trap.

## 5. Python reference

```python
def nn_layernorm(X, gamma, beta, shift):
    """X: (rows, cols) int8; gamma: (cols,) int8; beta: (cols,) int32."""
    n  = cols
    s1 = X.sum(axis=1);  s2 = (X * X).sum(axis=1)
    V  = n * s2 - s1 * s1;  q = isqrt(V) per row (math.isqrt);  q = max(q, 1)
    D  = n * X - s1;  N = D * gamma
    t  = sign(N) * ((2 * |N| + q) // (2 * q))
    return sat8(rshift(t + beta, shift))
```

`math.isqrt` is the exact floor square root, the same value the bit-by-bit loop gives.
All of it is int64, so it is exact.

## 6. Tests

`gamma` is random int8 (both signs) and `beta` random within `±2¹²` unless stated;
`shift` is chosen so most outputs land inside int8.

**Default:**

| Case | rows × cols | Notes |
|---|---|---|
| one element | 1 × 1 | `V = 0`: output is `rshift(beta)` |
| constant rows | 20 × 30 | every row constant: `V = 0`, output `beta` |
| smallest spread | 20 × 30 | each row all equal except one element ±1: `V = n − 1`, `q` small, `z` large |
| one outlier | 16 × 64 | zeros and one 127: `z ≈ √n`, saturates |
| largest spread | 16 × 64 | rows alternating −128 / 127 |
| variance a small perfect square | 20 × 4 | rows `[a, a, a+2, a+2]`: `V = 16`, `q = 4` exactly, so an `isqrt` that misses exact squares shows |
| edges | 17 × 33 | partial tiles both ways; padded columns must not count |
| partial block | 5 × 20 | 5 of 32 lanes active |
| exactly one block | 32 × 16 | |
| many blocks | 300 × 20 | 10 blocks over 4 SMs |
| long rows | 3 × 4000 | |
| model-sized rows | 40 × 384 | `gamma` ≈ 64, small `beta` |
| shift 0 / 31 | 20 × 40 | ends of the shift range |
| views | 20 × 40 | `ldt_x ≠ ldt_y` |
| in place | 20 × 40 | `Y == X` |
| launch splitting | 45 × 10, `-DNN_MAX_ROWS_PER_LAUNCH=10` | 5 launches |
| invalid | `rows = 0`; `cols = 0`; `cols = 32768`; `gamma = NULL`; `beta = NULL`; `ldt_x` too small; `shift = 32`; `X == Y` with different `ldt` | return -1, `Y` untouched |

**Checks** as in softmax.md §6: `Y` starts all poison and every non-element byte must stay
poison; `X`'s padding holds random junk, so a sum that strays into padding shows. stderr
must be empty. Default cases also run on the superscalar host at widths 1/2/4/10.

**Large (`--large`, scalar host only):** 1024 × 384; 2048 × 1536; and 2,097,200 × 1 in
place (more rows than one launch holds).

**`nn_isqrt64` on its own** (`isqrt` in `run_nn_tests.py`, test program
`tests/t_isqrt.c`, run on the host CPU). Layernorm cases can't reach `V ≥ 2⁴²`: that needs
rows of more than ~16,000 columns, and a row-wise block that long exceeds the GPU's issue
cap. So the helper is checked directly against Python's `math.isqrt` on about 250
values: 0 – 17, `k² − 1, k², k² + 1` for chosen and random `k` up to `2²² − 1`, values
around `2³²` and `2⁴²`, `2⁴⁴ − 1`, and 100 random values below `2⁴⁴`.

## 7. Cost

- **Kernel (measured, `-O0`):** pass 1 is 43 warp instructions per column, pass 2 is 146,
  so **189 per column** for 32 rows at once. `isqrt` adds 22 × 65 = 1,430 once per row
  (per warp), about 1,590 per row in all. The only branches are the split guard, the two
  column loops and the `isqrt` loop (a constant 22 steps); no library calls.
- **Estimate:** with full warps on all 4 SMs,

  ```
  device cycles ≈ ceil(rows / 32) / 4 × (189 × cols + ~1,590)  ≈ 1.5 × rows × cols
  ```

  for rows of a few hundred columns or more (the per-row part is about 2% at 384
  columns).
- **Measured** (whole-program cycles, scalar host):

  | Case | Model | Measured |
  |---|---|---|
  | 32 × 16 vs 1 × 1 | 15 columns × 189 = 2,835 more | 4,683 − 1,848 = 2,835 |
  | 1024 × 384 | 8 × (189 × 384 + 1,590) = 593,328 | 593,371 |
  | 2048 × 1536 | 16 × (189 × 1536 + 1,590) = 4,670,304 | 4,670,236 |

- **Issue cap:** a block (one warp) issues about `189 · cols + 1,590`, so rows up to about
  5,200 columns fit the per-block cap of 1,000,000 (softmax.md §3).

## 8. Correctness notes

- One thread per row, each writing only its own row: block and warp order don't matter.
- Every intermediate is exact within its type (§2), so the Python reference matches bit
  for bit.
- The result is symmetric: negating every `x` in a row negates every `t` (the rounding
  is symmetric too), as it is for real layer normalization.

## 9. Limitations and next steps

- **No epsilon parameter** (§1). If a quantization scheme ever has rows whose variance
  is comparable to ε, an `eps` term can be added to `V` (in `n²`·input units).
- **`isqrt` floors.** Rounding `q` to nearest would halve the worst-case relative error
  (`1/q` → `1/(2q)`) at the cost of a few instructions per row.
- **One thread per row**, as softmax.md §9: SMs idle when `rows` is small.
- **Cost at `-O0`**: see lut8.md §9.
