// Scalar multiply by 4, T lanes, no divergence (matches
// simulation/kernels.py's SCALAR_MULTIPLY_KERNEL / the reduced CUDA
// example: N == threadsPerBlock, so every lane owns exactly one element
// and the `if (idx < n)` boundary check is always true).
#include <stdint.h>
#include "../simt_isa.h"

#define VEC_LEN 4
#define SCALAR 4
#define RESULT_BASE 0x100000

static int32_t vec[VEC_LEN] = {2, 3, 5, 7};

int main(void) {
    // T (THREADS_PER_WARP) is 32, not VEC_LEN -- narrow to exactly the
    // lanes this kernel actually uses instead of relying on an `if`
    // boundary check (the real CUDA idiom), since an ordinary C `if` here
    // would compile to a genuinely data-dependent branch that lanes 4..31
    // disagree on, which SIMT-uniform BRANCH doesn't allow (see
    // docs/CUDA/plan.md) -- TMC is the right tool for a static launch size.
    simt_tmc(VEC_LEN);
    uint32_t idx = simt_tid() & 0xFF;
    int32_t* out = (int32_t*)RESULT_BASE;
    out[idx] = vec[idx] * SCALAR;
    return 0;
}
