// vit_baseline.c -- the CPU-only baseline for vit.c: the same integer
// arithmetic in plain C on the host CPU, no NPU or GPU, so the two outputs
// should be identical and only the cycle counts differ. run_vit.py runs both.
//
// Step 1: LayerNorm of VIT_NUM_BATCHES batches (VIT_ROWS x VIT_DIM, stacked)
// with no learned weights.
//
// Tensors are plain row-major int8 arrays. Compiled -O2.
#include <stdint.h>
#include "nn.h"          // nn_cycles() only
#include "vit.h"

// The input, row-major (make_input.py).
static const int8_t X[] = {
#include VIT_INPUT_FILE
};
_Static_assert(sizeof X == VIT_ROWS * VIT_DIM, "the input file doesn't match vit.h's sizes");

// Read by run_vit.py after the run: the output (row-major), its size, and
// the cycles spent in the model.
int8_t         vit_out[VIT_ROWS * VIT_DIM];
const uint32_t vit_out_rows = VIT_ROWS;
const uint32_t vit_out_cols = VIT_DIM;
uint64_t       vit_cycles;

// Floor square root of v, bit by bit.
static uint32_t isqrt64(uint64_t v) {
    uint32_t q = 0;
    for (int k = 31; k >= 0; k--) {
        uint32_t cand = q | (1u << k);
        if ((uint64_t)cand * cand <= v)
            q = cand;
    }
    return q;
}

static int8_t sat8(int32_t v) {
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Each row: z = (x - mean) / std, output round(z * VIT_LN_ONE), saturated to
// int8. In integers, with n = cols, s1 = sum x, s2 = sum x^2:
//   n * std = q = isqrt(n * s2 - s1^2)   (1 for a constant row)
//   z * VIT_LN_ONE = (n * x - s1) * VIT_LN_ONE / q, rounded half away from zero
// -- nn_layernorm's arithmetic (docs/design/layernorm.md) with gamma = 1.0,
// beta = 0 and shift = 0.
static void layernorm(const int8_t* X, int8_t* Y, uint32_t rows, uint32_t cols) {
    int32_t n = (int32_t)cols;
    for (uint32_t r = 0; r < rows; r++) {
        const int8_t* x = X + r * cols;
        int8_t*       y = Y + r * cols;

        int32_t s1 = 0;
        int32_t s2 = 0;
        for (uint32_t c = 0; c < cols; c++) {
            s1 += x[c];
            s2 += x[c] * x[c];
        }
        uint64_t V = (uint64_t)n * (uint32_t)s2 - (uint64_t)((int64_t)s1 * s1);
        uint32_t q = isqrt64(V);
        if (q == 0)
            q = 1;

        for (uint32_t c = 0; c < cols; c++) {
            int32_t num = (n * x[c] - s1) * VIT_LN_ONE;
            int32_t t;
            if (num >= 0)
                t = (int32_t)((2u * (uint32_t)num + q) / (2u * q));
            else
                t = -(int32_t)((2u * (uint32_t)(-num) + q) / (2u * q));
            y[c] = sat8(t);
        }
    }
}

int main(void) {
    uint64_t t0 = nn_cycles();
    layernorm(X, vit_out, VIT_ROWS, VIT_DIM);
    uint64_t t1 = nn_cycles();

    vit_cycles = t1 - t0;
    return 0;
}
