// Tests MUL, MULH, MULHU, MULHSU: multiplying two 32-bit values whose true
// product needs more than 32 bits, across all four signedness combinations.
//
// This version uses portable C only (no inline asm), so which instructions
// GCC actually emits depends on -O level: -O0/-O1 lower the 64-bit multiply
// generically using just mul/mulhu, while -O2/-O3 recognize the widening-
// multiply idiom and emit mulh/mulhu/mulhsu directly. Run this through
// run_opt_tests.py (which sweeps -O0..-O3) to get coverage of all four
// instructions. See mul_high_test_asm.c for a version that forces the exact
// instructions via inline asm regardless of optimization level.
#include <stdint.h>

int compute() {
    // volatile: without this, GCC constant-folds the entire computation at
    // -O1+ (both operands are literals), collapsing it to a single `li` of
    // the final answer -- no multiply instructions would run at all.
    volatile int32_t a_s = -100000;
    volatile int32_t b_s = 60000;
    uint32_t a_u = (uint32_t)a_s;
    uint32_t b_u = (uint32_t)b_s;

    int64_t  ss    = (int64_t)a_s * (int64_t)b_s;
    uint64_t uu    = (uint64_t)a_u * (uint64_t)b_u;
    int64_t  su    = (int64_t)a_s * (int64_t)(uint64_t)b_u;

    int32_t  lo    = (int32_t)(ss & 0xFFFFFFFFu);
    int32_t  ss_hi = (int32_t)(ss >> 32);
    uint32_t uu_hi = (uint32_t)(uu >> 32);
    int32_t  su_hi = (int32_t)(su >> 32);

    // Combine the four partial products into one checksum so a single
    // return register (a0) can confirm all of them are bit-exact.
    uint32_t checksum = (uint32_t)lo * 1u
                       + (uint32_t)ss_hi * 3u
                       + uu_hi * 5u
                       + (uint32_t)su_hi * 7u;
    return (int)checksum;
}

#ifdef NATIVE_TEST
#include <stdio.h>
int main() {
    printf("%d\n", compute());
    return 0;
}
#else
int main() {
    volatile int result = compute();
    return result;
}
#endif
