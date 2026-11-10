// vit_baseline.c -- the CPU-only baseline for vit.c: the same integer
// arithmetic in plain C on the host CPU, no NPU or GPU, so the two outputs
// should be identical and only the cycle counts differ. run_vit.py runs both.
//
// The model so far, on VIT_ROWS x VIT_DIM activations (VIT_NUM_BATCHES
// batches stacked):
//   1. XN  = LayerNorm(X), no learned weights           VIT_ROWS x VIT_DIM
//   2. QKV = XN . Wqkv, no bias                          VIT_ROWS x VIT_QKV
//   3. S   = Q_h . K_h^T / 8 for every batch and head     VIT_S_ROWS x VIT_BATCH
//      (one 32 x 32 matrix per (batch b, head h), stored as slot b * 6 + h)
//   4. P   = softmax of every row of S                   VIT_S_ROWS x VIT_BATCH
//   5. O   = P_h . V_h for every batch and head, the     VIT_ROWS x VIT_DIM
//      heads side by side (head h = columns h * 64 ..)
//   6. A   = O . Wo, no bias                              VIT_ROWS x VIT_DIM
//   7. X1  = X + A (first residual)                       VIT_ROWS x VIT_DIM
//   8. XN2 = LayerNorm(X1), no learned weights            VIT_ROWS x VIT_DIM
//   9. H   = XN2 . W1, no bias (MLP first layer)          VIT_ROWS x VIT_HIDDEN
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

static const int8_t Wo[] = {
#include "input/row_major/Wo_384_384.txt"
};
_Static_assert(sizeof Wo == VIT_DIM * VIT_DIM, "the Wo file doesn't match vit.h's sizes");

static const int8_t W1[] = {
#include "input/row_major/W1_384_1536.txt"
};
_Static_assert(sizeof W1 == VIT_DIM * VIT_HIDDEN, "the W1 file doesn't match vit.h's sizes");

static int8_t XN[VIT_ROWS * VIT_DIM];    // LayerNorm output
static const uint16_t exp_table[] = {
#include "input/exp_table_256.txt"
};
_Static_assert(sizeof exp_table == 256 * sizeof(uint16_t), "the exp table must have 256 values");

static int8_t QKV[VIT_ROWS * VIT_QKV];   // Q | K | V
static int8_t S[VIT_S_ROWS * VIT_BATCH]; // scores
static int8_t P[VIT_S_ROWS * VIT_BATCH]; // probabilities
static int8_t O[VIT_ROWS * VIT_DIM];     // head outputs, side by side
static int8_t A[VIT_ROWS * VIT_DIM];     // output projection
static int8_t X1[VIT_ROWS * VIT_DIM];    // first residual
static int8_t XN2[VIT_ROWS * VIT_DIM];   // LN2's output

// Read by run_vit.py after the run: the output (row-major), its size, and
// the cycles of the whole model and of each step.
int8_t         vit_out[VIT_ROWS * VIT_HIDDEN];       // FC1's output H
const uint32_t vit_out_rows = VIT_ROWS;
const uint32_t vit_out_cols = VIT_HIDDEN;
uint64_t       vit_cycles;
uint64_t       vit_cycles_layernorm1;
uint64_t       vit_cycles_qkv;
uint64_t       vit_cycles_scores;
uint64_t       vit_cycles_softmax;
uint64_t       vit_cycles_pv;
uint64_t       vit_cycles_proj;
uint64_t       vit_cycles_residual1;
uint64_t       vit_cycles_layernorm2;
uint64_t       vit_cycles_fc1;

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

// S = sat8(rshift(Q_h . K_h^T, shift)) for every batch and head. Q_h of
// batch b is rows b * 32 .. b * 32 + 31, columns h * 64 .. h * 64 + 63 of
// QKV; K_h the same rows, columns VIT_DIM + h * 64 ... K^T is never formed:
// S(i, j) is the dot product of row i of Q_h and row j of K_h.
static void attention_scores(const int8_t* QKV, int8_t* S, uint32_t shift) {
    for (uint32_t b = 0; b < VIT_NUM_BATCHES; b++) {
        for (uint32_t h = 0; h < VIT_HEADS; h++) {
            const int8_t* q = QKV + b * VIT_BATCH * VIT_QKV + h * VIT_HEAD_DIM;
            const int8_t* k = q + VIT_DIM;
            int8_t*       s = S + (b * VIT_HEADS + h) * VIT_BATCH * VIT_BATCH;
            for (uint32_t i = 0; i < VIT_BATCH; i++) {
                for (uint32_t j = 0; j < VIT_BATCH; j++) {
                    int32_t acc = 0;
                    for (uint32_t d = 0; d < VIT_HEAD_DIM; d++)
                        acc += q[i * VIT_QKV + d] * k[j * VIT_QKV + d];
                    s[i * VIT_BATCH + j] = sat8(rshift(acc, shift));
                }
            }
        }
    }
}

// Each row: m = max, e = table[m - s] (m - s is 0 .. 255), sum = sum of e,
// P = e * 128 / sum rounded half up, at most 127 -- nn_softmax's arithmetic
// (softmax.md).
static void softmax(const int8_t* S, int8_t* P, uint32_t rows, uint32_t cols,
                    const uint16_t* table) {
    for (uint32_t r = 0; r < rows; r++) {
        const int8_t* s = S + r * cols;
        int8_t*       p = P + r * cols;
        int32_t m = s[0];
        for (uint32_t c = 1; c < cols; c++)
            if (s[c] > m)
                m = s[c];
        uint32_t sum = 0;
        for (uint32_t c = 0; c < cols; c++)
            sum += table[m - s[c]];
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t y = (table[m - s[c]] * 128u + sum / 2u) / sum;
            p[c] = (int8_t)(y > 127u ? 127u : y);
        }
    }
}

// O = sat8(rshift(P_h . V_h, shift)) for every batch and head. P_h is slot
// b * 6 + h of P (32 x 32); V_h is the rows of batch b, columns
// 2 * VIT_DIM + h * 64 .. of QKV (32 x 64); the result goes to the same rows,
// columns h * 64 .. of O.
static void weighted_values(const int8_t* P, const int8_t* QKV, int8_t* O, uint32_t shift) {
    for (uint32_t b = 0; b < VIT_NUM_BATCHES; b++) {
        for (uint32_t h = 0; h < VIT_HEADS; h++) {
            const int8_t* p = P + (b * VIT_HEADS + h) * VIT_BATCH * VIT_BATCH;
            const int8_t* v = QKV + b * VIT_BATCH * VIT_QKV + 2 * VIT_DIM + h * VIT_HEAD_DIM;
            int8_t*       o = O + b * VIT_BATCH * VIT_DIM + h * VIT_HEAD_DIM;
            for (uint32_t i = 0; i < VIT_BATCH; i++) {
                for (uint32_t d = 0; d < VIT_HEAD_DIM; d++) {
                    int32_t acc = 0;
                    for (uint32_t j = 0; j < VIT_BATCH; j++)
                        acc += p[i * VIT_BATCH + j] * v[j * VIT_QKV + d];
                    o[i * VIT_DIM + d] = sat8(rshift(acc, shift));
                }
            }
        }
    }
}

// Y = sat8(rshift(A + B, shift)), element by element.
static void add(const int8_t* A, const int8_t* B, int8_t* Y, uint32_t n, uint32_t shift) {
    for (uint32_t i = 0; i < n; i++)
        Y[i] = sat8(rshift(A[i] + B[i], shift));
}

int main(void) {
    uint64_t t0 = nn_cycles();
    layernorm(X, XN, VIT_ROWS, VIT_DIM);
    uint64_t t1 = nn_cycles();
    matmul(XN, Wqkv, QKV, VIT_ROWS, VIT_DIM, VIT_QKV, VIT_QKV_SHIFT);
    uint64_t t2 = nn_cycles();
    attention_scores(QKV, S, VIT_SCORE_SHIFT);
    uint64_t t3 = nn_cycles();
    softmax(S, P, VIT_S_ROWS, VIT_BATCH, exp_table);
    uint64_t t4 = nn_cycles();
    weighted_values(P, QKV, O, VIT_PV_SHIFT);
    uint64_t t5 = nn_cycles();
    matmul(O, Wo, A, VIT_ROWS, VIT_DIM, VIT_DIM, VIT_PROJ_SHIFT);
    uint64_t t6 = nn_cycles();
    add(X, A, X1, VIT_ROWS * VIT_DIM, VIT_RES_SHIFT);
    uint64_t t7 = nn_cycles();
    layernorm(X1, XN2, VIT_ROWS, VIT_DIM);
    uint64_t t8 = nn_cycles();
    matmul(XN2, W1, vit_out, VIT_ROWS, VIT_DIM, VIT_HIDDEN, VIT_FC1_SHIFT);
    uint64_t t9 = nn_cycles();

    vit_cycles_layernorm1 = t1 - t0;
    vit_cycles_qkv       = t2 - t1;
    vit_cycles_scores    = t3 - t2;
    vit_cycles_softmax   = t4 - t3;
    vit_cycles_pv        = t5 - t4;
    vit_cycles_proj      = t6 - t5;
    vit_cycles_residual1 = t7 - t6;
    vit_cycles_layernorm2 = t8 - t7;
    vit_cycles_fc1       = t9 - t8;
    vit_cycles           = t9 - t0;
    return 0;
}
