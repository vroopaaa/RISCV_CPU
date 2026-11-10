# Lego: lookup table (`nn_lut8`) — GPU

Applies any function of one int8 value to every element of a tensor, by table lookup:

```
Y(r, c) = table[X(r, c) + 128]
```

`X` and `Y` are `rows × cols` int8 tensors in the library's padded tile layout. `table`
is 256 int8 values with `table[i] = f(i − 128)`, so `table[0]` is `f(-128)` and
`table[255]` is `f(127)`. The lego does not know what `f` is. Any element-wise
nonlinearity whose input and output are int8 (GELU, sigmoid, tanh, h-swish, exp for
softmax, ...) is one table, built offline in Python with whatever input and output
scales the caller wants.

This is a **GPU lego**, and the first one: it also fixes the pattern the later GPU legos
follow (kernel + wrapper in one file, launch shape, launch splitting, how the test
driver builds and checks a GPU lego).

Shared rules (number format, tile layout, lego conventions, GPU kernel rules, testing)
are in [00-overview.md](00-overview.md).

## 1. Operation

For `0 ≤ r < rows`, `0 ≤ c < cols`:

```
Y(r, c) = table[X(r, c) + 128]
```

Nothing else is written. Padding elements of `Y` keep whatever they held before the
call, which is zero under the overview's padding rule. This matters because many
functions are non-zero at 0 (sigmoid(0) = 0.5): if the lego applied `f` to padding, the
next matrix multiply would sum garbage.

**Building a table** (the caller's job, not part of the lego). If an int8 value `x`
stands for the real number `x · s_in`, and the output should stand for `y · s_out`:

```
table[i] = sat8(round(f((i − 128) · s_in) / s_out))      for i = 0 … 255
```

The table is a constant, generated in Python like the weights.

## 2. Interface

```c
int nn_lut8(const int8_t* X, uint32_t ldt_x,
            int8_t*       Y, uint32_t ldt_y,
            uint32_t rows, uint32_t cols,
            const int8_t* table);
```

| Parameter | Meaning |
|---|---|
| `X`, `ldt_x` | input tensor view and its tiles per tile-row |
| `Y`, `ldt_y` | output tensor view and its tiles per tile-row |
| `rows`, `cols` | real tensor size (not padded) |
| `table` | 256 int8 values, `table[i] = f(i − 128)` |

Returns 0, or -1 without launching anything if any check fails:
`rows, cols ≥ 1`; `X, Y, table` non-null; `ldt_x ≥ NN_PAD16(cols)/16`;
`ldt_y ≥ NN_PAD16(cols)/16`; if `X == Y` then `ldt_x == ldt_y`.

- **In place is allowed:** `Y == X` with the same `ldt`. Each thread reads one element
  and then writes that same element, and no two threads touch the same element, so no
  thread can read a value another thread has already replaced. Any other overlap
  between `X`, `Y` and `table` is not allowed. That is the caller's responsibility and
  is not checked.
- **No alignment requirement.** The lego reads and writes single bytes. (Tensors start
  on tile boundaries anyway, overview §2.)
- `rows · cols` must fit in 32 bits. Any tensor that fits in RAM does
  (64 MB < 2³²), so this is not checked.

## 3. How it runs on the GPU

One thread per **real** element: `total = rows · cols` threads, numbered in row-major
order of the real (unpadded) matrix. Thread `e` handles element
`(r, c) = (e / cols, e % cols)`.

**Device kernel** (`lut8_kernel`):

```
i = simt_global_id()
if simt_split(i < count):                  # last block may be partly unused
    e = base + i
    r = e / cols                           # divu: one instruction, no branch
    c = e % cols                           # remu
    x = X[nn_off(r, c, ldt_x)]             # signed byte load
    Y[nn_off(r, c, ldt_y)] = table[x + 128]
simt_join()
```

- **No branch depends on lane data** except the `simt_split` guard (overview §4). The
  index `x + 128` is always in 0 … 255, so there is no bounds check. `cols`, `ldt_x`,
  `ldt_y` and `base` come from the args struct and are the same in every lane.
- **The result leaves through memory** (the store to `Y`) before `simt_join()`.
- In hardware `x + 128` costs nothing: on an 8-bit value it just flips bit 7, which turns
  the signed value into an unsigned table index.

**Host wrapper** (`nn_lut8`):

```
check parameters (§2), return -1 on failure
total = rows · cols
for base = 0; base < total; base += NN_MAX_OUTPUTS_PER_LAUNCH:
    count = min(total − base, NN_MAX_OUTPUTS_PER_LAUNCH)
    args = { X, ldt_x, Y, ldt_y, cols, base, count, table }
    simt_launch(lut8_kernel, NN_THREADS, nn_blocks(count), &args)
return 0
```

- One static `args` struct, reused for every launch. `simt_launch` doesn't return until
  the whole grid is done (overview §3).
- `NN_THREADS` = 128 threads per block; `nn_blocks(count) = ceil(count / 128)`.
- **Launch splitting:** one launch covers at most 65,535 blocks × 128 =
  8,388,480 elements. A bigger tensor is split into several launches that differ only in
  `base` and `count`.

**Why one thread per real element, not one per padded element.** Threads over the
padded buffer would avoid the divide, but then the padding threads would need a second
guard (`r < rows && c < cols`) and still compute their position from the tile index,
which needs a divide by the number of tile columns anyway. Counting only real elements
gives one guard and a dense thread range. In this model `divu`/`remu` are one warp
instruction each, like any other.

## 4. Additions to `nn.h`

The overview lists these as `nn.h` contents; this lego is the first that needs them:

```c
#define NN_THREADS 128u                       // threads per block (4 warps × 32)
#ifndef NN_MAX_OUTPUTS_PER_LAUNCH
#define NN_MAX_OUTPUTS_PER_LAUNCH (65535u * NN_THREADS)
#endif
static inline __attribute__((always_inline)) uint32_t nn_blocks(uint32_t total);
static inline __attribute__((always_inline)) uint32_t nn_off(uint32_t r, uint32_t c, uint32_t ldt);
int nn_lut8(...);                             // §2
```

`nn_off` and `nn_blocks` are `always_inline` because kernels call them at `-O0`
(overview §4). `NN_MAX_OUTPUTS_PER_LAUNCH` can be overridden with `-D` so a test can
force launch splitting with a small tensor.

## 5. Python reference

```python
def nn_lut8(X, table):
    """X: (rows, cols) int8. table: 256 int8 values, table[i] = f(i - 128).
    Returns (rows, cols) int8: table[X + 128]."""
```

Plain row-major arrays; the driver converts to and from the tile layout.

## 6. Tests

**Tables.** Most cases use a **random** table. A smooth function like GELU would hide
an off-by-one index (`table[x + 127]` is almost the same value), while a random table
turns any wrong index into a wrong output. Two cases use special tables:
- **identity** (`table[i] = i − 128`): output must equal input;
- **constant 77**: `f(0) ≠ 0`, the case where writing padding would be visible.

**Default:**

| Case | rows × cols | Notes |
|---|---|---|
| one element | 1 × 1 | one thread in one block |
| one full tile | 16 × 16 | 256 threads = 2 full blocks |
| edges | 17 × 33 | partial tiles in both directions |
| partial block | 5 × 27 | 135 elements: 2 blocks, the second with 7 active threads |
| exactly one block | 8 × 16 | 128 threads |
| many blocks | 40 × 100 | 32 blocks, more than the 4 SMs, so round-robin |
| one row, long | 1 × 1000 | `r` always 0 |
| one column, long | 1000 × 1 | `c` always 0 |
| identity table | 20 × 40 | Y = X |
| constant table | 20 × 40 | padding of `Y` must be unchanged |
| views | `X` a column slice of a wider buffer, `Y` written into a slice of another | `ldt > needed`, and `ldt_x ≠ ldt_y` so mixing them up shows; tiles outside the `Y` view must be unchanged |
| in place | 20 × 40, `Y == X` | |
| launch splitting | 30 × 33 with `-DNN_MAX_OUTPUTS_PER_LAUNCH=300` | 990 elements in 4 launches (300, 300, 300, 90) |
| invalid | `rows = 0`; `cols = 0`; `table = NULL`; `ldt_x` too small; `X == Y` with different `ldt` | return -1, `Y` untouched |

**Checks** (overview §7), with one difference from the NPU legos: the driver fills the
whole `Y` buffer, padding included, with a poison byte before the run. Afterwards the
real elements must match the reference, and **every other byte, including `Y`'s
padding, must still be poison**. That proves the lego never writes padding. `X`'s
padding and the area outside the `X` view are filled with random junk, so reading from a
wrong position gives a wrong output. stderr must be empty (no SM fault, no divergence
assert, no timeout).

**Large (`--large`):** 256 × 4096 (1,048,576 elements, 8,192 blocks); 1000 × 1000
(edges, 7,813 blocks); 2048 × 4100 (8,396,800 elements, more than the 8,388,480 one
launch can hold, so the wrapper must really split). That last case is the only one that
catches a wrapper that forgets to cap `count`: with the `-D` override in the default
set, an uncapped launch just redoes correct work, while past the real limit the 16-bit
block field overflows.

**Driver changes** (`run_nn_tests.py`):
- **Per-file optimisation level.** Lego kernel files get their own `-O` (GPU legos
  `-O0`), separate from the test program's (`-O2` default). Each source is compiled to an
  object file with its own level, then linked.
- **Per-case extra flags** (e.g. `-DNN_MAX_OUTPUTS_PER_LAUNCH=300`).
- **Host modes.** Default cases run on the scalar host and on the superscalar host at
  widths 1/2/4/10, like GEMM: the wrapper's launch loop runs on the host. Large cases
  run on the scalar host only, since the device cycles don't depend on the host mode and
  each run is slow.

## 7. Cost

- **Device cycles** are issued warp instructions on the busiest SM (overview §8). A
  warp issues the whole kernel once, however many of its lanes are active.
- **Kernel length (measured):** `lut8_kernel` compiled at `-O0` is **112
  instructions** with one branch (the split). Most of it is `-O0` stack traffic: 43 loads
  and 18 stores, against 1 load from `X`, 1 from `table` and 1 store to `Y`.
  Every warp issues all 112.
- **Estimate:** `total` elements = `total / 32` warps, shared by 4 SMs, so

  ```
  device cycles ≈ ceil(total / 128) / 4 × 4 warps × 112 ≈ 0.875 × total
  ```

  plus a few host instructions per launch.
- **Measured** (whole-program cycles, scalar host):

  | Tensor | Elements | Model `0.875 × total` | Measured |
  |---|---|---|---|
  | 256 × 4096 | 1,048,576 | 917,504 | 917,635 |
  | 1000 × 1000 | 1,000,000 | 875,000 | 875,523 |
  | 2048 × 4100 (2 launches) | 8,396,800 | 7,347,200 | 7,347,855 |

  The difference is host start-up and the wrapper, about 130 cycles per launch.
- **Example:** a 256 × 512 tensor (131,072 elements) costs about 115,000 device cycles.
  The GEMM that produces a tensor that size, 256 × 512 × 512, takes 122,957 cycles on the
  scalar host and 18,631 at issue width 10 (gemm.md §6). So applying a table to a matmul's
  output would cost **about 6× the matmul itself** on the fast host. Element-wise work on
  the GPU at `-O0` is not cheap next to the NPU; see §9.

## 8. Correctness notes

- Every thread writes a different element (the mapping `e → (r, c)` is one-to-one on
  `0 ≤ e < total`), so the result doesn't depend on block or warp order.
- No lane ever touches padding, so `Y`'s padding stays zero if it was zero, and the
  table's value at 0 never matters.
- Results don't depend on anything left over from earlier calls: the only state is the
  args struct, which the wrapper fills completely before every launch.

## 9. Limitations and next steps

- **Cost at -O0** (§7). Ways to bring it down, all later decisions:
  - write the kernel body in assembly (about 15 instructions instead of 112);
  - **fuse the table into the NPU output stage**: a 256-byte table applied after
    `store_c8`'s clamp, which would make an activation after a matmul almost free. This
    is a hardware change and would need its own design file;
  - more SMs or wider warps.
- **Memory model.** Every lane reads its own table entry. With no memory model that's
  free; with one, the table would live in per-SM fast memory or a small hardware LUT.
- **Functions of one value only.** Anything that combines two tensors (a residual add,
  multiplying by a gate computed elsewhere) is not covered: that is the element-wise
  lego, not a table. A function of one value, however complicated (e.g. SiLU,
  `x · sigmoid(x)`), is still a single table.
