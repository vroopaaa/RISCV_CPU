// vit_baseline.c -- the CPU-only baseline for vit.c: the same integer
// arithmetic in plain C on the host CPU, no NPU or GPU, so the two outputs
// should be identical and only the cycle counts differ. run_vit.py runs both.
//
// The model so far, on VIT_ROWS x VIT_DIM activations (VIT_NUM_BATCHES
// batches stacked):
//   1. XN  = LayerNorm(X), no learned weights           VIT_ROWS x VIT_DIM
//   2. QKV = XN . Wqkv, no bias                          VIT_ROWS x VIT_QKV
//
// Tensors are plain row-major int8 arrays. Compiled -O2.
#include <stdint.h>
#include "nn.h"          // nn_cycles() only
#include "vit.h"

// The input and the weights, row-major (make_input.py).
static const int8_t X[] = {
#include VIT_X_ROW_MAJOR
};
_Static_assert(sizeof X == VIT_ROWS * VIT_DIM, "the X file doesn't match vit.h's sizes");

static const int8_t Wqkv[] = {
#include "input/row_major/Wqkv_384_1152.txt"
};
_Static_assert(sizeof Wqkv == VIT_DIM * VIT_QKV, "the Wqkv file doesn't match vit.h's sizes");

static int8_t XN[VIT_ROWS * VIT_DIM];    // LayerNorm output

// Read by run_vit.py after the run: the output (row-major), its size, and
// the cycles of the whole model and of each step.
int8_t         vit_out[VIT_ROWS * VIT_QKV];
const uint32_t vit_out_rows = VIT_ROWS;
const uint32_t vit_out_cols = VIT_QKV;
uint64_t       vit_cycles;
uint64_t       vit_cycles_layernorm;
uint64_t       vit_cycles_qkv;

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

// Rounding right shift, halves toward +infinity (nn.h's nn_rshift).
static int32_t rshift(int32_t v, uint32_t s) {
    if (s == 0)
        return v;
    return (v + (1 << (s - 1))) >> s;
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

// Y = sat8(rshift(X . W, shift)), X: M x K, W: K x N, Y: M x N.
static void matmul(const int8_t* X, const int8_t* W, int8_t* Y,
                   uint32_t M, uint32_t K, uint32_t N, uint32_t shift) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; k++)
                acc += X[i * K + k] * W[k * N + j];
            Y[i * N + j] = sat8(rshift(acc, shift));
        }
    }
}

int main(void) {
    uint64_t t0 = nn_cycles();
    layernorm(X, XN, VIT_ROWS, VIT_DIM);
    uint64_t t1 = nn_cycles();
    matmul(XN, Wqkv, vit_out, VIT_ROWS, VIT_DIM, VIT_QKV, VIT_QKV_SHIFT);
    uint64_t t2 = nn_cycles();

    vit_cycles_layernorm = t1 - t0;
    vit_cycles_qkv       = t2 - t1;
    vit_cycles           = t2 - t0;
    return 0;
}
