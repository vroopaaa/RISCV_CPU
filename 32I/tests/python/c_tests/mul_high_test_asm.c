// Tests MUL, MULH, MULHU, MULHSU: multiplying two 32-bit values whose true
// product needs more than 32 bits, across all four signedness combinations.
//
// This version drives the four instructions directly via inline asm, so the
// exact instructions used are fixed regardless of optimization level -- unlike
// mul_high_test.c (portable C), whose emitted instructions depend on -O level.
#include <stdint.h>

#if defined(__riscv)
static inline int32_t rv_mul(int32_t a, int32_t b) {
    int32_t r;
    __asm__ volatile("mul %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline int32_t rv_mulh(int32_t a, int32_t b) {
    int32_t r;
    __asm__ volatile("mulh %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline uint32_t rv_mulhu(uint32_t a, uint32_t b) {
    uint32_t r;
    __asm__ volatile("mulhu %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline int32_t rv_mulhsu(int32_t a, uint32_t b) {
    int32_t r;
    __asm__ volatile("mulhsu %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
#endif

int compute() {
    int32_t a_s = -100000;
    int32_t b_s = 60000;
    uint32_t a_u = (uint32_t)a_s;
    uint32_t b_u = (uint32_t)b_s;

#if defined(__riscv)
    int32_t  lo    = rv_mul(a_s, b_s);      // MUL:     low 32 bits (signedness-agnostic)
    int32_t  ss_hi = rv_mulh(a_s, b_s);     // MULH:    high 32 bits, signed * signed
    uint32_t uu_hi = rv_mulhu(a_u, b_u);    // MULHU:   high 32 bits, unsigned * unsigned
    int32_t  su_hi = rv_mulhsu(a_s, b_u);   // MULHSU:  high 32 bits, signed * unsigned
#else
    int64_t  ss    = (int64_t)a_s * (int64_t)b_s;
    uint64_t uu    = (uint64_t)a_u * (uint64_t)b_u;
    int64_t  su    = (int64_t)a_s * (int64_t)(uint64_t)b_u;

    int32_t  lo    = (int32_t)(ss & 0xFFFFFFFFu);
    int32_t  ss_hi = (int32_t)(ss >> 32);
    uint32_t uu_hi = (uint32_t)(uu >> 32);
    int32_t  su_hi = (int32_t)(su >> 32);
#endif

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
