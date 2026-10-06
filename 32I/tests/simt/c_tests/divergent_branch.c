// Manufactured divergent branch (matches simulation/kernels.py's
// DIVERGENT_BRANCH_KERNEL / 32I/tests/simt/regfile+exec coverage before
// this file existed): if (idx < 2) result = 100; else result = 200;
// mem[idx] = result. An ordinary C `if` compiles to a real conditional
// branch, so each half is wrapped in its own simt_split()/simt_join() --
// by the time the `if` runs, tmask has already been narrowed to exactly
// the lanes where the condition is true, so the compiled branch is
// trivially uniform among the surviving active lanes (see
// docs/CUDA/plan.md's BRANCH-must-be-uniform rule).
#include <stdint.h>
#include "../simt_isa.h"

#define RESULT_BASE 0x100000

int main(void) {
    uint32_t idx = simt_tid() & 0xFF;
    uint32_t pred = (idx < 2);
    int32_t result = 0;

    simt_split(pred);
    if (pred) {
        result = 100;
    }
    simt_join();

    uint32_t not_pred = !pred;
    simt_split(not_pred);
    if (not_pred) {
        result = 200;
    }
    simt_join();

    int32_t* out = (int32_t*)RESULT_BASE;
    out[idx] = result;
    return 0;
}
