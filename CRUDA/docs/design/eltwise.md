# Lego: element-wise add and multiply (`nn_add`, `nn_mul`) — GPU

Combines two tensors of the same shape element by element:

```
nn_add:  Y = sat8(rshift(A · 2^sa + B · 2^sb, shift))
nn_mul:  Y = sat8(rshift(A · B, shift))
```

`A`, `B` and `Y` are `rows × cols` int8 tensors in the library's padded tile layout.
`sa` and `sb` line the two inputs up on a common scale before the add, and `shift`
requantizes the int32 result back to int8 with the overview's rounding rule. This is the
library's version of cuBLAS `geam` (`C = α·A + β·B`) and cuDNN `OpTensor`, with the
scale factors restricted to powers of two.

This is a **GPU lego** and follows the pattern [lut8.md](lut8.md) set: one thread per
real element, kernel and host wrapper in one file, the same launch shape and launch
splitting, padding never touched. Shared rules are in [00-overview.md](00-overview.md).

## 1. Operation

For `0 ≤ r < rows`, `0 ≤ c < cols`, with `a = A(r, c)`, `b = B(r, c)`:

```
nn_add:  v = a · 2^sa + b · 2^sb            (int32)
nn_mul:  v = a · b                          (int32)
         Y(r, c) = sat8(rshift(v, shift))   (overview §1)
```

Nothing else is written; padding of `Y` keeps its value (zero).

**Why the scale shifts.** An int8 value only means something together with its scale.
Say `A` holds values in steps of 2⁻⁴ (`a` stands for `a/16`), `B` in steps of 2⁻⁶, and
the result should be in steps of 2⁻⁵. Adding the raw integers `a + b` would add
sixteenths to sixty-fourths. Instead:

| Step | Integer | Real value |
|---|---|---|
| inputs | `a = 20`, `b = 40` | 20/16 = 1.25, 40/64 = 0.625 |
| put both in 2⁻⁶ steps: `sa = 2`, `sb = 0` | `20·4 + 40 = 120` | 120/64 = 1.875 ✓ |
| to 2⁻⁵ steps: `shift = 1` | `rshift(120, 1) = 60` | 60/32 = 1.875 ✓ |

For a multiply the scales add instead: 2⁻⁴ × 2⁻⁶ = 2⁻¹⁰, so a result wanted in 2⁻⁷
steps uses `shift = 3`. Choosing `sa`, `sb` and `shift` is the caller's job (the
network's quantization plan); the lego just applies them.

## 2. Interface

```c
int nn_add(const int8_t* A, uint32_t ldt_a, uint32_t sa,
           const int8_t* B, uint32_t ldt_b, uint32_t sb,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift);

int nn_mul(const int8_t* A, uint32_t ldt_a,
           const int8_t* B, uint32_t ldt_b,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift);
```

| Parameter | Meaning |
|---|---|
| `A`, `ldt_a` / `B`, `ldt_b` | input tensor views and their tiles per tile-row |
| `sa`, `sb` | (`nn_add` only) left shifts of `A` and `B`, 0 – 16 |
| `Y`, `ldt_y` | output tensor view and its tiles per tile-row |
| `rows`, `cols` | real tensor size, the same for all three |
| `shift` | requantization shift, 0 – 31 |

Return 0, or -1 without launching anything if a check fails: `rows, cols ≥ 1`;
`A, B, Y` non-null; `ldt_a`, `ldt_b`, `ldt_y` each `≥ NN_PAD16(cols)/16`;
`sa, sb ≤ 16`; `shift ≤ 31`; if `Y == A` then `ldt_y == ldt_a`; if `Y == B` then
`ldt_y == ldt_b`.

- **In place is allowed:** `Y == A` (`x += r`, the residual add) or `Y == B`, with the
  same `ldt` as that operand. Each thread reads its element of both inputs and then writes
  that same element, so no thread reads a value another thread has replaced (same
  argument as lut8.md §2).
- **`A == B` is allowed**, with any `ldt` (both are only read): `nn_mul(A, A)` squares,
  `nn_add(A, A)` doubles.
- Any other overlap is not allowed; the caller's responsibility, not checked.
- **Why `sa, sb ≤ 16`:** then `|v| ≤ 2 · 128 · 2¹⁶ = 2²⁴`, far inside int32, including
  the rounding constant `rshift` adds. Real scale differences are a few bits, so 16 is
  plenty. For `nn_mul`, `|v| ≤ 2¹⁴` always.

## 3. How it runs on the GPU

Same mapping as lut8: thread `e` handles `(r, c) = (e / cols, e % cols)` of the real
matrix, guarded by `simt_split(i < count)`.

**One kernel for both operations**, `eltwise_kernel`, with the operation in the args
struct:

```
i = simt_global_id()
if simt_split(i < count):
    e = base + i;  r = e / cols;  c = e % cols
    a = A[nn_off(r, c, ldt_a)]
    b = B[nn_off(r, c, ldt_b)]
    if op == ADD:  v = a · (1 << sa) + b · (1 << sb)
    else:          v = a · b
    Y[nn_off(r, c, ldt_y)] = nn_sat8(nn_rshift(v, shift))
simt_join()
```

- **The `op` branch is allowed.** It depends only on the args struct, so every lane
  takes the same direction (overview §4 forbids branches on *lane* data). Two separate
  kernels would save that branch (about 3 instructions of 190) but duplicate the rest.
- **`a · (1 << sa)`, not `a << sa`.** In C, shifting a negative number left is undefined
  behaviour; multiplying by a power of two gives the same value and is defined. In
  hardware it is still a shift.
- **`nn_rshift` and `nn_sat8` are branch-free** (they run on lane data):

  ```c
  nn_rshift(v, s) = (v + ((1u << s) >> 1)) >> s     // s = 0: adds 0, shifts 0 -> v
  nn_sat8(v):       hi = (v > 127);   v = v + hi * (127 - v);
                    lo = (v < -128);  v = v + lo * (-128 - v);
  ```

  `(1u << s) >> 1` is `2^(s-1)` for `s ≥ 1` and 0 for `s = 0`, so the overview's
  two-case `rshift` needs no `if`. The comparisons compile to `slt`/`sgt` (a 0/1
  value), not branches; the draft kernel's disassembly confirms it. `>>` on a negative
  `int32_t` is an arithmetic shift in gcc (RISC-V `sra`), which is what the rounding rule
  needs. The NPU does the same steps in hardware (its output stage); on the GPU they are
  software.

**Host wrappers.** `nn_add` and `nn_mul` check their parameters (§2), fill one static
args struct (`op`, pointers, `ldt`s, `cols`, `sa`, `sb`, `shift`), and run the same
launch loop as `nn_lut8`: launches of at most `NN_MAX_OUTPUTS_PER_LAUNCH` elements,
`NN_THREADS` = 128 threads per block.

## 4. Additions to `nn.h`

`nn_rshift` and `nn_sat8` (`static inline`, `always_inline`, as above; the overview
already lists them as `nn.h` helpers), and the `nn_add` / `nn_mul` prototypes.

## 5. Python reference

```python
def nn_add(A, B, sa, sb, shift):   # sat8(rshift(A * 2**sa + B * 2**sb, shift))
def nn_mul(A, B, shift):           # sat8(rshift(A * B, shift))
```

In int64, with the existing `rshift` / `sat8`; plain row-major arrays.

## 6. Tests

Every case runs for **both** `nn_add` and `nn_mul` unless it says otherwise. Inputs are
random int8 unless stated.

**Default:**

| Case | rows × cols | Notes |
|---|---|---|
| one element | 1 × 1 | |
| edges | 17 × 33 | partial tiles |
| partial block | 5 × 27 | 135 elements, second block 7 threads |
| many blocks | 40 × 100 | 32 blocks over 4 SMs |
| scales (add) | 20 × 40 | `sa = 2, sb = 0`, `sa = 0, sb = 3`, `sa = sb = 16` |
| shift 0 | 20 × 40 | outputs saturate a lot; checks the `s = 0` path of `nn_rshift` |
| shift 31 | 20 × 40 | `|v| ≤ 2²⁴`, so every output rounds to 0; checks the far end of the shift range |
| rounding | 16 × 16 | inputs chosen so `v` lands exactly on halves, both signs, `shift` 1 and 2 |
| saturation | 16 × 16 | all `-128`: add gives `-256 → -128`; mul gives `16384 → 127` at `shift 0` |
| clamp boundary | 2 × 10 | `v` exactly 125 … 129 and −126 … −130 at `shift 0`, so a clamp edge off by one shows |
| views | 20 × 40 | `A`, `B`, `Y` slices of wider buffers, three different `ldt`s |
| in place, `Y == A` | 20 × 40 | |
| in place, `Y == B` | 20 × 40 | |
| `A == B` | 20 × 40 | square / double |
| launch splitting | 30 × 33, `-DNN_MAX_OUTPUTS_PER_LAUNCH=300` | 4 launches |
| invalid | `rows = 0`; `cols = 0`; `B = NULL`; `ldt_b` too small; `sa = 17`, `sb = 17` (add); `shift = 32`; `Y == A` with different `ldt`; `Y == B` with different `ldt` | return -1, `Y` untouched |

**Checks** as in lut8.md §6: the whole `Y` buffer starts poisoned, and every byte that is
not a real element must still be poison afterwards; padding and the area outside the
`A` / `B` views hold random junk, so reading a wrong position shows. stderr must be
empty. Default cases run on the scalar host and superscalar widths 1/2/4/10.

**Large (`--large`, scalar host only):** 256 × 4096 for both operations, and 2048 × 4100
(two launches) for `nn_add`.

## 7. Cost

- **Kernel length (measured):** `eltwise_kernel` at `-O0` is 194 instructions; a warp
  issues **190 for add and 183 for mul** (the `op` branch skips the other side's 11 or 4
  instructions). That is ~1.7× lut8's 112: two input loads, the shifts, `nn_rshift` and
  `nn_sat8`. Its only branches are the split guard, the `op` branch and the jump that
  ends the add side; `nn_rshift` and `nn_sat8` compile without branches.
- **Estimate:** `device cycles ≈ total / 128 × 190 ≈ 1.5 × total` for add
  (`× 183 ≈ 1.43 × total` for mul).
- **Measured** (whole-program cycles, scalar host):

  | Case | Elements | Model | Measured |
  |---|---|---|---|
  | add, 256 × 4096 | 1,048,576 | 2,048 blocks per SM × 4 × 190 = 1,556,480 | 1,556,725 |
  | mul, 256 × 4096 | 1,048,576 | 2,048 × 4 × 183 = 1,499,136 | 1,499,371 |
  | add, 2048 × 4100 (2 launches) | 8,396,800 | (16,384 + 17) × 4 × 190 = 12,464,760 | 12,465,101 |

  The difference is host start-up and the wrappers, a few hundred cycles.
- **Example:** a residual add on a 256 × 512 tensor (131,072 elements) ≈ 195,000 device
  cycles, against 18,631 for the 256 × 512 × 512 GEMM at issue width 10. As with lut8,
  separate element-wise passes on the GPU at `-O0` cost much more than the matmul; see
  lut8.md §9 for the options (assembly kernels, fusing into the NPU output stage).

## 8. Correctness notes

- Each thread writes one distinct element, so block and warp order don't matter.
- No lane reads or writes padding.
- `v` never overflows int32 (§2), so the result is exact before the shift and the
  Python reference matches bit for bit.

## 9. Limitations and next steps

- **No broadcasting.** `B` must be a full `rows × cols` tensor. Scaling every row or
  column by one value (per-channel scales, e.g. squeeze-excite) needs a broadcast form;
  add it when the first lego that needs it is designed.
- **Power-of-two scales only.** A general multiplier per input (`a · ma + b · mb`) can be
  added later as extra parameters, as the overview plans for requantization in general.
- **Only add and multiply.** Subtract is `nn_add` with a negated operand (or a later
  `op`); min / max are not needed yet.
- **Cost at `-O0`** (§7).
