// Element-wise add / multiply lego on the GPU -- docs/design/eltwise.md.
//
//   nn_add:  Y = sat8(rshift(A * 2^sa + B * 2^sb, shift))
//   nn_mul:  Y = sat8(rshift(A * B, shift))
//
// Same pattern as lut8.c: one thread per real element, thread e handles
// (e / cols, e % cols), padding is never read or written, and tensors bigger
// than NN_MAX_OUTPUTS_PER_LAUNCH take several launches. One kernel serves
// both operations; op is the same in every lane, so branching on it is
// allowed. This file contains SIMT code, so it is compiled at -O0.
#include "../include/nn.h"
#include "simt_isa.h"

#define ELT_ADD 0u
#define ELT_MUL 1u

struct eltwise_args {
    uint32_t      op;      // ELT_ADD or ELT_MUL
    const int8_t* A;
    uint32_t      ldt_a;
    uint32_t      sa;      // add only
    const int8_t* B;
    uint32_t      ldt_b;
    uint32_t      sb;      // add only
    int8_t*       Y;
    uint32_t      ldt_y;
    uint32_t      cols;
    uint32_t      shift;
    uint32_t      base;    // first element of this launch
    uint32_t      count;   // elements in this launch
};

void eltwise_kernel(void* p) {
    const struct eltwise_args* g = (const struct eltwise_args*)p;
    uint32_t i = simt_global_id();
    if (simt_split(i < g->count)) {          // the last block may be partly unused
        uint32_t e = g->base + i;
        uint32_t r = e / g->cols;
        uint32_t c = e % g->cols;
        int32_t  a = g->A[nn_off(r, c, g->ldt_a)];
        int32_t  b = g->B[nn_off(r, c, g->ldt_b)];
        int32_t  v;
        if (g->op == ELT_ADD)
            v = a * (int32_t)(1u << g->sa) + b * (int32_t)(1u << g->sb);   // a << sa is undefined for a < 0
        else
            v = a * b;
        g->Y[nn_off(r, c, g->ldt_y)] = (int8_t)nn_sat8(nn_rshift(v, g->shift));
    }
    simt_join();
}

// One args struct for every launch: simt_launch returns only once the whole
// grid has finished, so the next launch can't overwrite it while in use.
static struct eltwise_args eltwise_args;

// Shared by nn_add and nn_mul once their parameters are checked: fill the
// args struct and launch, split into pieces of NN_MAX_OUTPUTS_PER_LAUNCH.
static void eltwise_run(uint32_t op,
                        const int8_t* A, uint32_t ldt_a, uint32_t sa,
                        const int8_t* B, uint32_t ldt_b, uint32_t sb,
                        int8_t* Y, uint32_t ldt_y,
                        uint32_t rows, uint32_t cols, uint32_t shift) {
    uint32_t total = rows * cols;
    for (uint32_t base = 0; base < total; base += NN_MAX_OUTPUTS_PER_LAUNCH) {
        uint32_t count = total - base;
        if (count > NN_MAX_OUTPUTS_PER_LAUNCH) count = NN_MAX_OUTPUTS_PER_LAUNCH;
        eltwise_args.op    = op;
        eltwise_args.A     = A;
        eltwise_args.ldt_a = ldt_a;
        eltwise_args.sa    = sa;
        eltwise_args.B     = B;
        eltwise_args.ldt_b = ldt_b;
        eltwise_args.sb    = sb;
        eltwise_args.Y     = Y;
        eltwise_args.ldt_y = ldt_y;
        eltwise_args.cols  = cols;
        eltwise_args.shift = shift;
        eltwise_args.base  = base;
        eltwise_args.count = count;
        simt_launch(eltwise_kernel, NN_THREADS, nn_blocks(count), &eltwise_args);
    }
}

// Checks shared by both wrappers (eltwise.md section 2, except sa / sb).
static int eltwise_params_ok(const int8_t* A, uint32_t ldt_a,
                             const int8_t* B, uint32_t ldt_b,
                             const int8_t* Y, uint32_t ldt_y,
                             uint32_t rows, uint32_t cols, uint32_t shift) {
    if (rows == 0 || cols == 0) return 0;
    if (A == 0 || B == 0 || Y == 0) return 0;
    uint32_t ct = NN_PAD16(cols) / NN_TILE;
    if (ldt_a < ct || ldt_b < ct || ldt_y < ct) return 0;
    if (shift > 31) return 0;
    if (Y == A && ldt_y != ldt_a) return 0;
    if (Y == B && ldt_y != ldt_b) return 0;
    return 1;
}

int nn_add(const int8_t* A, uint32_t ldt_a, uint32_t sa,
           const int8_t* B, uint32_t ldt_b, uint32_t sb,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift) {
    if (!eltwise_params_ok(A, ldt_a, B, ldt_b, Y, ldt_y, rows, cols, shift)) return -1;
    if (sa > 16 || sb > 16) return -1;
    eltwise_run(ELT_ADD, A, ldt_a, sa, B, ldt_b, sb, Y, ldt_y, rows, cols, shift);
    return 0;
}

int nn_mul(const int8_t* A, uint32_t ldt_a,
           const int8_t* B, uint32_t ldt_b,
           int8_t*       Y, uint32_t ldt_y,
           uint32_t rows, uint32_t cols, uint32_t shift) {
    if (!eltwise_params_ok(A, ldt_a, B, ldt_b, Y, ldt_y, rows, cols, shift)) return -1;
    eltwise_run(ELT_MUL, A, ldt_a, 0, B, ldt_b, 0, Y, ldt_y, rows, cols, shift);
    return 0;
}
