// Softmax lego on the GPU, row-wise -- docs/design/softmax.md.
//
//   m = max_c X(r, c);  e = table[m - X(r, c)];  S = sum_c e
//   Y(r, c) = min(127, (e * 128 + S / 2) / S)
//
// One thread per row, three passes over the row's real columns (max, sum,
// divide and write). Padding is neither read nor written. Blocks are one
// warp (NN_ROW_THREADS) so a few hundred rows still use every SM; tables
// bigger than NN_MAX_ROWS_PER_LAUNCH rows take several launches that differ
// only in base and count. This file contains SIMT code, so it is compiled
// at -O0.
#include "../include/nn.h"
#include "simt_isa.h"

struct softmax_args {
    const int8_t*   X;
    uint32_t        ldt_x;
    int8_t*         Y;
    uint32_t        ldt_y;
    uint32_t        cols;
    uint32_t        base;    // first row of this launch
    uint32_t        count;   // rows in this launch
    const uint16_t* table;
};

void softmax_kernel(void* p) {
    const struct softmax_args* g = (const struct softmax_args*)p;
    uint32_t i = simt_global_id();
    if (simt_split(i < g->count)) {          // the last block may be partly unused
        uint32_t r = g->base + i;

        // Pass 1: row maximum. Branch-free: (x > m) is 0 or 1.
        int32_t m = -128;
        for (uint32_t c = 0; c < g->cols; c++) {
            int32_t x  = g->X[nn_off(r, c, g->ldt_x)];
            int32_t gt = x > m;
            m = m + gt * (x - m);
        }

        // Pass 2: row sum of e = table[m - x]; m - x is always 0 ... 255.
        uint32_t s = 0;
        for (uint32_t c = 0; c < g->cols; c++)
            s += g->table[m - g->X[nn_off(r, c, g->ldt_x)]];

        // Pass 3: e * 128 / S rounded half up. e <= S, so y <= 128 and
        // subtracting (y > 127) is the whole clamp.
        for (uint32_t c = 0; c < g->cols; c++) {
            uint32_t e = g->table[m - g->X[nn_off(r, c, g->ldt_x)]];
            uint32_t y = (e * 128u + s / 2u) / s;
            y = y - (y > 127u);
            g->Y[nn_off(r, c, g->ldt_y)] = (int8_t)y;
        }
    }
    simt_join();
}

// One args struct for every launch: simt_launch returns only once the whole
// grid has finished, so the next launch can't overwrite it while in use.
static struct softmax_args softmax_args;

int nn_softmax(const int8_t*   X, uint32_t ldt_x,
               int8_t*         Y, uint32_t ldt_y,
               uint32_t rows, uint32_t cols,
               const uint16_t* table) {
    if (rows == 0 || cols == 0 || cols > 65535) return -1;
    if (X == 0 || Y == 0 || table == 0) return -1;
    if (table[0] == 0) return -1;            // S >= table[0] must be >= 1: no divide by 0
    uint32_t ct = NN_PAD16(cols) / NN_TILE;
    if (ldt_x < ct || ldt_y < ct) return -1;
    if (X == Y && ldt_x != ldt_y) return -1;

    for (uint32_t base = 0; base < rows; base += NN_MAX_ROWS_PER_LAUNCH) {
        uint32_t count = rows - base;
        if (count > NN_MAX_ROWS_PER_LAUNCH) count = NN_MAX_ROWS_PER_LAUNCH;
        softmax_args.X     = X;
        softmax_args.ldt_x = ldt_x;
        softmax_args.Y     = Y;
        softmax_args.ldt_y = ldt_y;
        softmax_args.cols  = cols;
        softmax_args.base  = base;
        softmax_args.count = count;
        softmax_args.table = table;
        simt_launch(softmax_kernel, NN_ROW_THREADS, nn_row_blocks(count), &softmax_args);
    }
    return 0;
}
