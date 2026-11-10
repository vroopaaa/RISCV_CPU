// nn.h -- the lego library: shared definitions and every lego's host wrapper.
// Rules (number format, tile layout, conventions) are in docs/design/00-overview.md;
// each lego has its own design file next to it.
//
// Every tensor is a matrix stored as padded 16 x 16 tiles (256 contiguous
// bytes each, row-major inside the tile, tiles in row-major order) and is
// passed as a pointer to its first tile plus ldt, the number of tiles in one
// tile-row of the underlying buffer. Padding is always zero.
#ifndef NN_H
#define NN_H

#include <stdint.h>

#define NN_TILE         16u
#define NN_TILE_BYTES   (NN_TILE * NN_TILE)
#define NN_PAD16(x)     (((x) + NN_TILE - 1u) & ~(NN_TILE - 1u))

// GPU launch shape (overview section 3): blocks of NN_THREADS threads, and at
// most NN_MAX_OUTPUTS_PER_LAUNCH outputs per launch (the block count is a
// 16-bit field). Overridable with -D so tests can force launch splitting.
#define NN_THREADS      128u
#ifndef NN_MAX_OUTPUTS_PER_LAUNCH
#define NN_MAX_OUTPUTS_PER_LAUNCH (65535u * NN_THREADS)
#endif

// Row-wise GPU legos (one thread per row) use one-warp blocks, so a few
// hundred rows still spread over every SM (docs/design/softmax.md section 3).
#define NN_ROW_THREADS  32u
#ifndef NN_MAX_ROWS_PER_LAUNCH
#define NN_MAX_ROWS_PER_LAUNCH (65535u * NN_ROW_THREADS)
#endif

// The helpers below are always_inline because GPU kernels call them at -O0
// (overview section 4).

// Blocks needed for `total` outputs: ceil(total / NN_THREADS).
static inline __attribute__((always_inline)) uint32_t nn_blocks(uint32_t total) {
    return (total + NN_THREADS - 1u) / NN_THREADS;
}

// Blocks needed for `rows` rows of a row-wise lego: ceil(rows / NN_ROW_THREADS).
static inline __attribute__((always_inline)) uint32_t nn_row_blocks(uint32_t rows) {
    return (rows + NN_ROW_THREADS - 1u) / NN_ROW_THREADS;
}

// Byte offset of element (r, c) in a tiled tensor with ldt tiles per tile-row.
static inline __attribute__((always_inline)) uint32_t nn_off(uint32_t r, uint32_t c, uint32_t ldt) {
    return ((r >> 4) * ldt + (c >> 4)) * NN_TILE_BYTES + (r & 15u) * NN_TILE + (c & 15u);
}

// Requantization helpers (overview section 1), branch-free because GPU
// kernels apply them to per-lane data (overview section 4).

// Rounding right shift, halves toward +inf, 0 <= s <= 31. (1u << s) >> 1 is
// 2^(s-1) for s >= 1 and 0 for s = 0, so s = 0 needs no special case.
// >> on a negative int32_t is an arithmetic shift in gcc (RISC-V sra).
static inline __attribute__((always_inline)) int32_t nn_rshift(int32_t v, uint32_t s) {
    return (v + (int32_t)((1u << s) >> 1)) >> s;
}

// Floor square root of the 64-bit value vhi:vlo, which must be < 2^44 (the
// result has at most NN_ISQRT64_BITS = 22 bits). Bit by bit (restoring square
// root): one compare and one OR per result bit. Branch-free: cand^2 <= V is
// compared on 32-bit halves, because gcc turns a 64-bit <= into branches.
// The 64-bit multiply compiles inline (mul + mulhu), with no library call.
#define NN_ISQRT64_BITS 22u
static inline __attribute__((always_inline)) uint32_t nn_isqrt64(uint32_t vhi, uint32_t vlo) {
    uint32_t q = 0;
    for (uint32_t k = 0; k < NN_ISQRT64_BITS; k++) {
        uint32_t bit  = 1u << (NN_ISQRT64_BITS - 1u - k);
        uint32_t cand = q | bit;
        uint64_t sq   = (uint64_t)cand * cand;
        uint32_t hi   = (uint32_t)(sq >> 32);
        uint32_t lo   = (uint32_t)sq;
        uint32_t le   = (hi < vhi) | ((hi == vhi) & (lo <= vlo));
        q = q | (le * bit);
    }
    return q;
}

// Clamp to [-128, 127]. The comparisons give 0/1 values, not branches.
static inline __attribute__((always_inline)) int32_t nn_sat8(int32_t v) {
    int32_t hi = v > 127;
    v = v + hi * (127 - v);
    int32_t lo = v < -128;
    v = v + lo * (-128 - v);
    return v;
}

// The host CPU's cycle counter (docs/design/cycle_counter.md): cycles
// completed so far, including every LAUNCH's device cycles. Host code only:
// an SM lane faults on rdcycle. The two halves are separate instructions, so
// if the low word wraps between them the high word is read again and the
// pair retried.
static inline __attribute__((always_inline)) uint64_t nn_cycles(void) {
    uint32_t hi, lo, hi2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(hi));
        __asm__ volatile ("rdcycle %0" : "=r"(lo));
        __asm__ volatile ("rdcycleh %0" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

// GEMM on the NPU (docs/design/gemm.md):
//   Y = sat8(rshift(X . W + bias, shift))
// X is M x K, W is K x N, Y is M x N; bias is NN_PAD16(N) int32 or NULL.
// Returns 0, or -1 without touching the NPU if a parameter is invalid.
int nn_gemm(const int8_t*  X, uint32_t ldt_x,
            const int8_t*  W, uint32_t ldt_w,
            const int32_t* bias,
            int8_t*        Y, uint32_t ldt_y,
            uint32_t M, uint32_t K, uint32_t N,
            uint32_t shift);

// GEMM with a transposed weight on the NPU (docs/design/gemm_bt.md):
//   Y = sat8(rshift(X . Wt^T + bias, shift))
// Wt is stored N x K (ldt_wt >= NN_PAD16(K)/16); the NPU loads its tiles with
// load_b_t. Everything else as nn_gemm.
int nn_gemm_bt(const int8_t*  X,  uint32_t ldt_x,
               const int8_t*  Wt, uint32_t ldt_wt,
               const int32_t* bias,
               int8_t*        Y,  uint32_t ldt_y,
               uint32_t M, uint32_t K, uint32_t N,
               uint32_t shift);

// Lookup table on the GPU (docs/design/lut8.md):
//   Y(r, c) = table[X(r, c) + 128]      for every real element; padding untouched
// table is 256 int8 values, table[i] = f(i - 128). Y == X (same ldt) is allowed.
// Returns 0, or -1 without launching anything if a parameter is invalid.
int nn_lut8(const int8_t* X, uint32_t ldt_x,
            int8_t*       Y, uint32_t ldt_y,
            uint32_t rows, uint32_t cols,
            const int8_t* table);

// Element-wise add / multiply on the GPU (docs/design/eltwise.md):
//   nn_add: Y = sat8(rshift(A * 2^sa + B * 2^sb, shift))    0 <= sa, sb <= 16
//   nn_mul: Y = sat8(rshift(A * B, shift))
// for every real element; padding untouched. Y == A or Y == B (same ldt as
// that operand) is allowed, and so is A == B. shift 0 - 31.
// Return 0, or -1 without launching anything if a parameter is invalid.
int nn_add(const int8_t* A, uint32_t ldt_a, uint32_t sa,
           const int8_t* B, uint32_t ldt_b, uint32_t sb,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift);

int nn_mul(const int8_t* A, uint32_t ldt_a,
           const int8_t* B, uint32_t ldt_b,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift);

// Softmax along each row on the GPU (docs/design/softmax.md):
//   m = row max;  e = table[m - X];  S = row sum of e;
//   Y = min(127, (e * 128 + S / 2) / S)        probability in steps of 1/128
// table is 256 uint16 values (table[d] = round(2^15 * exp(-d * s_in)) for a
// real softmax), table[0] >= 1. Padding is neither read nor written.
// 1 <= cols <= 65535; Y == X (same ldt) is allowed.
// Returns 0, or -1 without launching anything if a parameter is invalid.
int nn_softmax(const int8_t*   X, uint32_t ldt_x,
               int8_t*         Y, uint32_t ldt_y,
               uint32_t rows, uint32_t cols,
               const uint16_t* table);

// Layer normalization along each row on the GPU (docs/design/layernorm.md):
//   s1 = sum x;  s2 = sum x^2;  q = isqrt(n * s2 - s1^2) (1 if 0);  n = cols
//   t  = round((n * x - s1) * gamma[c] / q)      (half away from zero)
//   Y  = sat8(rshift(t + beta[c], shift))
// gamma: cols int8, beta: cols int32 (|beta| <= 2^29). Padding is neither read
// nor written. 1 <= cols <= 32767; shift 0 - 31; Y == X (same ldt) is allowed.
// Returns 0, or -1 without launching anything if a parameter is invalid.
int nn_layernorm(const int8_t*  X, uint32_t ldt_x,
                 int8_t*        Y, uint32_t ldt_y,
                 uint32_t rows, uint32_t cols,
                 const int8_t*  gamma,
                 const int32_t* beta,
                 uint32_t shift);

#endif // NN_H
