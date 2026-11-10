// Lookup-table lego on the GPU -- docs/design/lut8.md.
//
//   Y(r, c) = table[X(r, c) + 128]      for every real element
//
// One thread per real element, numbered in row-major order of the real
// (unpadded) matrix: thread e handles (e / cols, e % cols). Padding is never
// read or written. The host wrapper splits tensors bigger than
// NN_MAX_OUTPUTS_PER_LAUNCH into several launches that differ only in base
// and count. This file contains SIMT code, so it is compiled at -O0.
#include "../include/nn.h"
#include "simt_isa.h"

struct lut8_args {
    const int8_t* X;
    uint32_t      ldt_x;
    int8_t*       Y;
    uint32_t      ldt_y;
    uint32_t      cols;
    uint32_t      base;    // first element of this launch
    uint32_t      count;   // elements in this launch
    const int8_t* table;
};

void lut8_kernel(void* p) {
    const struct lut8_args* a = (const struct lut8_args*)p;
    uint32_t i = simt_global_id();
    if (simt_split(i < a->count)) {          // the last block may be partly unused
        uint32_t e = a->base + i;
        uint32_t r = e / a->cols;
        uint32_t c = e % a->cols;
        int32_t  x = a->X[nn_off(r, c, a->ldt_x)];
        a->Y[nn_off(r, c, a->ldt_y)] = a->table[x + 128];
    }
    simt_join();
}

// One args struct for every launch: simt_launch returns only once the whole
// grid has finished, so the next launch can't overwrite it while in use.
static struct lut8_args lut8_args;

int nn_lut8(const int8_t* X, uint32_t ldt_x,
            int8_t*       Y, uint32_t ldt_y,
            uint32_t rows, uint32_t cols,
            const int8_t* table) {
    if (rows == 0 || cols == 0) return -1;
    if (X == 0 || Y == 0 || table == 0) return -1;
    uint32_t ct = NN_PAD16(cols) / NN_TILE;
    if (ldt_x < ct || ldt_y < ct) return -1;
    if (X == Y && ldt_x != ldt_y) return -1;

    uint32_t total = rows * cols;
    for (uint32_t base = 0; base < total; base += NN_MAX_OUTPUTS_PER_LAUNCH) {
        uint32_t count = total - base;
        if (count > NN_MAX_OUTPUTS_PER_LAUNCH) count = NN_MAX_OUTPUTS_PER_LAUNCH;
        lut8_args.X     = X;
        lut8_args.ldt_x = ldt_x;
        lut8_args.Y     = Y;
        lut8_args.ldt_y = ldt_y;
        lut8_args.cols  = cols;
        lut8_args.base  = base;
        lut8_args.count = count;
        lut8_args.table = table;
        simt_launch(lut8_kernel, NN_THREADS, nn_blocks(count), &lut8_args);
    }
    return 0;
}
