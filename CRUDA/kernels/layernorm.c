// Layer normalization lego on the GPU, row-wise -- docs/design/layernorm.md.
//
//   s1 = sum x;  s2 = sum x^2;  V = n * s2 - s1^2;  q = isqrt(V), 1 if 0
//   D  = n * x - s1;  t = round(D * gamma[c] / q)  (half away from zero)
//   Y  = sat8(rshift(t + beta[c], shift))                      n = cols
//
// D / q = (x - mean) / std exactly, so the mean is never rounded. One thread
// per row, two passes over the row's real columns (sums, then normalize and
// write); padding is neither read nor written. Blocks are one warp
// (NN_ROW_THREADS) and tensors bigger than NN_MAX_ROWS_PER_LAUNCH rows take
// several launches, as in softmax.c. This file contains SIMT code, so it is
// compiled at -O0.
#include "../include/nn.h"
#include "simt_isa.h"

struct layernorm_args {
    const int8_t*  X;
    uint32_t       ldt_x;
    int8_t*        Y;
    uint32_t       ldt_y;
    uint32_t       cols;
    const int8_t*  gamma;
    const int32_t* beta;
    uint32_t       shift;
    uint32_t       base;    // first row of this launch
    uint32_t       count;   // rows in this launch
};

void layernorm_kernel(void* p) {
    const struct layernorm_args* g = (const struct layernorm_args*)p;
    uint32_t i = simt_global_id();
    if (simt_split(i < g->count)) {          // the last block may be partly unused
        uint32_t r = g->base + i;
        int32_t  n = (int32_t)g->cols;

        // Pass 1: sum and sum of squares.
        int32_t s1 = 0;
        int32_t s2 = 0;
        for (uint32_t c = 0; c < g->cols; c++) {
            int32_t x = g->X[nn_off(r, c, g->ldt_x)];
            s1 += x;
            s2 += x * x;
        }

        // V = n * s2 - s1^2 = n^2 * variance, in 64 bits (no 64-bit divide).
        // s1^2 is formed from |s1|, branch-free: sign = 1 - 2 * (s1 < 0).
        uint32_t abs_s1 = (uint32_t)(s1 * (1 - 2 * (s1 < 0)));
        uint64_t V = (uint64_t)(uint32_t)n * (uint32_t)s2 - (uint64_t)abs_s1 * abs_s1;
        uint32_t q = nn_isqrt64((uint32_t)(V >> 32), (uint32_t)V);   // V < 2^44 for cols <= 32767
        // Constant row: V = 0 and every D is 0, so t = 0 and the output is beta.
        // q = 0 would divide by zero below, which C leaves undefined (RISC-V
        // happens to return all ones and sg = 0 hides it; other ISAs trap).
        q = q + (q == 0);

        // Pass 2: t = round(D * gamma / q), half away from zero, then requantize.
        for (uint32_t c = 0; c < g->cols; c++) {
            int32_t  x   = g->X[nn_off(r, c, g->ldt_x)];
            int32_t  num = (n * x - s1) * g->gamma[c];
            int32_t  sg  = (num > 0) - (num < 0);
            uint32_t mag = (uint32_t)(num * sg);
            int32_t  t   = sg * (int32_t)((2u * mag + q) / (2u * q));
            g->Y[nn_off(r, c, g->ldt_y)] = (int8_t)nn_sat8(nn_rshift(t + g->beta[c], g->shift));
        }
    }
    simt_join();
}

// One args struct for every launch: simt_launch returns only once the whole
// grid has finished, so the next launch can't overwrite it while in use.
static struct layernorm_args layernorm_args;

int nn_layernorm(const int8_t*  X, uint32_t ldt_x,
                 int8_t*        Y, uint32_t ldt_y,
                 uint32_t rows, uint32_t cols,
                 const int8_t*  gamma,
                 const int32_t* beta,
                 uint32_t shift) {
    if (rows == 0 || cols == 0 || cols > 32767) return -1;
    if (X == 0 || Y == 0 || gamma == 0 || beta == 0) return -1;
    uint32_t ct = NN_PAD16(cols) / NN_TILE;
    if (ldt_x < ct || ldt_y < ct) return -1;
    if (shift > 31) return -1;
    if (X == Y && ldt_x != ldt_y) return -1;

    for (uint32_t base = 0; base < rows; base += NN_MAX_ROWS_PER_LAUNCH) {
        uint32_t count = rows - base;
        if (count > NN_MAX_ROWS_PER_LAUNCH) count = NN_MAX_ROWS_PER_LAUNCH;
        layernorm_args.X     = X;
        layernorm_args.ldt_x = ldt_x;
        layernorm_args.Y     = Y;
        layernorm_args.ldt_y = ldt_y;
        layernorm_args.cols  = cols;
        layernorm_args.gamma = gamma;
        layernorm_args.beta  = beta;
        layernorm_args.shift = shift;
        layernorm_args.base  = base;
        layernorm_args.count = count;
        simt_launch(layernorm_kernel, NN_ROW_THREADS, nn_row_blocks(count), &layernorm_args);
    }
    return 0;
}
