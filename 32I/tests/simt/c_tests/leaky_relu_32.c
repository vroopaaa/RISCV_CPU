// Leaky ReLU across a full 32-thread warp (1 SM) -- see docs/CUDA/plan.md /
// progress.md. Real RV32IM DIV (the M-extension, not a symbolic stand-in):
//   y[idx] = x[idx]       if x[idx] >= 0
//   y[idx] = x[idx] / 8   if x[idx] <  0   (truncating division)
#include <stdint.h>
#include "../simt_isa.h"

#define VEC_LEN 32
#define RESULT_BASE 0x100000

static int32_t x[VEC_LEN] = {
    -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
      0,   1,   2,   3,   4,   5,   6,   7,  8,  9, 10, 11, 12, 13, 14, 15,
};

int main(void) {
    uint32_t idx = simt_tid() & 0xFF;
    int32_t xi = x[idx];
    int32_t result = 0;

    uint32_t pred = (xi < 0);
    simt_split(pred);
    if (pred) {
        result = xi / 8;
    }
    simt_join();

    uint32_t not_pred = !pred;
    simt_split(not_pred);
    if (not_pred) {
        result = xi;
    }
    simt_join();

    int32_t* out = (int32_t*)RESULT_BASE;
    out[idx] = result;
    return 0;
}
