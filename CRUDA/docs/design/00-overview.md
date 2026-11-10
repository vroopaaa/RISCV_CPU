# Lego library: shared rules

The lego library is a set of small, general-purpose neural-network operations for the
32I system: a host CPU, a SIMT GPU and an NPU. Each lego is one operation (GEMM,
convolution, an activation, softmax, ...) with no knowledge of any particular network,
and each runs on the unit suited to it:

- **NPU legos** do matrix multiplication. The host CPU drives the NPU tile by tile with
  the custom-0 instructions.
- **GPU legos** do everything else. The host launches them on the SIMT GPU with
  `simt_launch`.

A network is a host program that calls legos in order. Building a different network
means writing a different host program, not new legos.

This file holds the rules every lego follows. Each lego's own design file covers only
what is specific to it, and refers back here for the rest.

## 1. Number format

- **Tensors between legos are int8.** Every lego reads and writes int8 tensors, except
  where its design file states otherwise (e.g. index outputs, int32 bias).
- **Accumulation is int32.** Products of int8 values are summed in a signed 32-bit
  accumulator. The caller must keep the true sum inside int32: with int8 × int8 products
  (|p| ≤ 16,384) that holds for any reduction length below 131,072. Overflow behaviour
  is not defined and is not tested.
- **Requantization (int32 → int8) is a rounding right shift, then a clamp:**

  ```
  rshift(v, s) = v                          if s == 0
               = (v + (1 << (s-1))) >> s    if 1 ≤ s ≤ 31   (arithmetic shift)
  sat8(v)      = min(max(v, -128), 127)
  out          = sat8(rshift(v, s))
  ```

  `rshift` rounds halves toward +∞ (e.g. rshift(-3, 1) = -1, rshift(3, 1) = 2).
  The shift amount `s` is a per-call parameter of each lego that requantizes. The NPU's
  output stage implements exactly this rule in hardware
  ([npu_output_stage.md](npu_output_stage.md)). A multiplier
  (`out = sat8(rshift(v · M, s))`) can be added later as an extra parameter without
  changing any interface that exists now.
- **Integer division** in C truncates toward zero. Where a lego divides, its design
  file states the exact rounding it uses, and the Python reference reproduces that rule
  (Python's `//` floors, so it is never used directly on values that can be negative).

## 2. Tensor layout: padded 16 × 16 tiles

Every tensor in the library is stored in one layout, the NPU's native one, so no lego
ever needs a layout conversion:

- **Every tensor is a 2-D matrix** of `rows × cols` int8 elements. A feature map
  `C × H × W` is the matrix `[C × H·W]`: element `(c, y, x)` is matrix element
  `(c, y·W + x)`.
- **Padded to whole tiles.** `rows_p = ceil16(rows)`, `cols_p = ceil16(cols)`. The
  buffer holds `rows_p · cols_p` bytes.
- **Tiles are 16 × 16 = 256 contiguous bytes**, row-major inside the tile, and tiles
  follow each other in row-major tile order.
- **Addressing.** A tensor is passed as a base pointer plus `ldt`, the number of tiles
  in one tile-row of the underlying buffer (normally `cols_p / 16`):

  ```
  off(r, c) = ((r >> 4) · ldt + (c >> 4)) · 256 + (r & 15) · 16 + (c & 15)
  ```

  `nn.h` provides this as `nn_off(r, c, ldt)`, plus `NN_PAD16(x)`.
- **Column slices.** Columns `16·t0 … 16·t0 + w − 1` of a wider tensor are the view
  `(base + t0 · 256, same ldt)`. A slice must start on a 16-column boundary. This
  replaces row strides: no lego takes an `ld…` element stride.
- **Padding is always zero.** A matrix multiply sums over the padded `K` columns, so
  non-zero padding would add garbage to real results. The rule that keeps it zero:
  - GPU legos write only real elements (`r < rows`, `c < cols`) and never touch padding;
  - NPU legos write whole tiles, but the output stage zeroes everything outside the real
    `M × N` region;
  - so a buffer's padding, once zero (`.bss` is zero at start), stays zero for good.

  This matters because many element-wise functions are non-zero at 0 (e.g. sigmoid).
  Applied to padding, they would break the next matrix multiply.
- **Alignment.** Every tensor starts on a 4-byte boundary (the NPU moves tiles as
  words).
- **Not tensors.** Bias (int32, one per output column, a plain array padded with zeros
  to `cols_p`), lookup tables, and index arrays are plain contiguous arrays. Their lego
  files state their layout.

## 3. Lego conventions

- **Every lego has a host wrapper** `int nn_<lego>(...)`, whichever unit it runs on. It
  checks its parameters and returns 0, or -1 without doing anything if a parameter is
  invalid. A caller never needs to know which unit a lego uses.
- **NPU legos** are plain host functions. They drive the NPU with the custom-0
  instructions and MMIO registers from `32I/tests/npu/npu_isa.h` (see
  [npu_output_stage.md](npu_output_stage.md)). No launch is involved; the work shows up
  as host cycles.
- **GPU legos** have two parts in one file:
  - the **device kernel** `void <lego>_kernel(void* p)`, which casts `p` to its args
    struct, computes `i = simt_global_id()` and guards all its work with
    `if (simt_split(i < total)) { ... } simt_join();`
  - the **host wrapper**, which fills the lego's args struct and calls `simt_launch`.
    A launch doesn't return until the whole grid has finished, so one static args
    struct per lego is safe to reuse on every call.
- **GPU launch shape.** One thread per output element unless the lego's file says
  otherwise (row-wise and single-thread legos exist where an output depends on a whole
  row or tensor). Blocks are `NN_THREADS` = 128 threads, the most a block can hold
  (4 warps × 32 lanes), and `total` outputs use `nn_blocks(total) = ceil(total / 128)`
  blocks.
- **Launch splitting.** The block count is a 16-bit field, so one launch covers at most
  65,535 × 128 = 8,388,480 outputs. A GPU lego whose output can be larger splits the
  work across several launches inside its wrapper. The limit is the macro
  `NN_MAX_OUTPUTS_PER_LAUNCH` in `nn.h`, overridable at build time so tests can
  exercise splitting with small shapes.
- **Optimisation levels are per file.** GPU lego files contain SIMT code, so they are
  compiled at `-O0` (`simt_isa.h` enforces this). NPU lego files and host programs
  contain none, so they are compiled at `-O2` by default (configurable in the driver),
  the same level the NPU superscalar analysis used. That keeps the host loops that drive
  the NPU realistic. `nn.h` therefore does not include `simt_isa.h`; only GPU lego files
  do.

## 4. Rules for GPU kernel code (from how the SIMT core works)

- **No branch may depend on per-lane data.** The SIMT core requires every lane of a warp
  to take the same direction at a branch, and stops with an assert if they disagree. So:
  - loop bounds come only from the args struct (all lanes run the same trip count);
  - choosing between two values (max, min, clamp, select) is written branch-free, using
    comparison results as 0/1 values and masks, never `if` or `?:` (at `-O0` gcc turns
    those into branches);
  - a genuinely divergent region goes inside `simt_split(...) ... simt_join()`.
  Every kernel's disassembly is checked for this before it is considered done.
- **Results leave a split region through memory.** A value computed inside a split
  region is stored before `simt_join()`, never read from a local variable afterwards.
- **Helpers called from kernels** (`nn_sat8`, `nn_rshift`, `nn_off`, ...) are
  `static inline __attribute__((always_inline))`, like the `simt_*` helpers, so a kernel
  is straight-line code inside one stack frame.
- **Each lane has a 1 KB stack.** Kernels keep only a few scalars as locals; no local
  arrays.
- **SM lanes cannot use the NPU.** Custom-0 instructions fault on a lane; only the host
  drives the NPU.

## 5. Files

```
SparseViT/
  include/nn.h        # NN_THREADS, NN_PAD16, nn_off, nn_blocks, nn_sat8, nn_rshift
                      #   + every lego's wrapper prototype
  kernels/<lego>.c    # one file per lego (NPU: host function; GPU: kernel + wrapper)
  ref/nn_ref.py       # Python reference: one function per lego, same integer
                      #   semantics, plus to_tiles / from_tiles layout helpers
  tests/t_<lego>.c    # one host test program per lego
  run_nn_tests.py     # test driver
  link.ld             # 64 MB RAM
  docs/design/        # this file + one design file per lego / hardware change
```

Programs are built with 32I's `tests/python/start.s`, include
`32I/tests/simt/simt_isa.h` and `32I/tests/npu/npu_isa.h`, and run on
`32I/build/run_test` (the same emulator `emul` uses).

## 6. Memory map (64 MB)

| Region | Address range | Contents |
|---|---|---|
| Program | `0x0000000` upward | `.text`, then `.data` (including the `.incbin` data block), then `.bss` (outputs, scratch) |
| Lane stacks | `0x3F70000` – `0x3FEFFFF` | 4 SMs × 128 lanes × 1 KB, set up by `GridLauncher` |
| Host stack | `0x3FF0000` – `0x3FFFFFF` | 64 KB (`CPU::HOST_STACK_RESERVE`) |
| NPU window | `0x80000000` + | memory-mapped NPU registers and tile buffers (not RAM) |

The program (code + data + bss) must end below `0x3F70000` (about 63.4 MB).
`link.ld` checks this with an `ASSERT`, so an oversized program fails at link time
instead of silently overlapping the stacks.

## 7. Testing

Every lego is tested on its own, bit-exact against its Python reference:

1. **Generate.** The driver draws seeded random int8 inputs and weights for one test
   shape and converts them to the padded tile layout (`to_tiles`). It writes them to
   `data.bin`, writes `data.h` with each tensor's offset, and computes the expected
   output with `nn_ref`.
2. **Build.** `start.s` + `tests/t_<lego>.c` + the lego file + `data.s` (one
   `.incbin "data.bin"` line), at `-O0`, linked with `SparseViT/link.ld`.
3. **Run** on `32I/build/run_test`, dumping the output buffer (address from `nm`).
4. **Check:**
   - the real elements must match the reference exactly (converted back with
     `from_tiles`); on failure the driver prints the first mismatching `(r, c)`, the
     expected value and the actual value;
   - **padding must be all zero**;
   - **guard bytes:** each output buffer is followed by a poisoned region, which must be
     unchanged afterwards (catches writes past the end of the buffer);
   - stderr must be empty (no GPU fault, no timeout, no out-of-range launch).

Each lego's file lists its test shapes in two sets:

- **default**: small and edge-case shapes, such as sizes that are not multiples of 16,
  totals that are not multiples of 128, a single element or row, and every mode or
  option the lego has. Runs in seconds.
- **large** (`--large`): big shapes that exercise long loops, many tiles and many
  blocks. Slow; run on demand.

Programs with NPU legos are also run on the superscalar host (`issue width` 1 / 2 / 4 /
10) and must give the same output.

## 8. Cycle reporting

- **What a cycle is here.** Host cycles are the CPU's `cycle_count` (one per
  instruction, or per issue group on the superscalar host). An NPU instruction counts
  like any other instruction. A GPU launch adds its device cycles: issued warp
  instructions on the busiest SM (see `GridLauncher`). These count instructions, not
  real timing.
- **Per-lego cycles.** A test or network program reads the host cycle counter before
  and after each lego call, and stores the difference in a results array that the
  driver dumps along with the outputs. This covers NPU legos (host cycles) and GPU legos
  (launch + device cycles) the same way.
  - The counter is read with the standard RISC-V `rdcycle` / `rdcycleh`, through
    `nn_cycles()` in `nn.h` (host code only). See [cycle_counter.md](cycle_counter.md).
- **Issue cap.** `GridLauncher` abandons a block after a fixed number of issued
  instructions (1,000,000 by default) to catch runaway kernels. Single-thread and
  row-wise GPU legos can legitimately exceed that, so `32I/tests/basic/main.cpp` raises
  it to 100,000,000.
- How a network's per-layer cycles are combined or extrapolated (e.g. one measured
  layer × repeat count) belongs in that network's own docs, not here.
