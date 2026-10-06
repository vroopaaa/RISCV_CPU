// Phase 1 unit test for SIMTCore's banked register file.
// Build/run:  make -C 32I run-simt-regfile-test
#include <cstdio>
#include <cstdlib>
#include "../../include/SIMT.h"
#include "../../include/memory.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    constexpr int T = SIMTCore::THREADS_PER_WARP, W = SIMTCore::WARPS_RESIDENT, R = SIMTCore::NUM_ARCH_REGS;
    Memory mem(4096); // unused by this register-file-only test; SIMTCore just needs a valid Memory*
    SIMTCore core(&mem);

    static_assert(SIMTCore::PHYS_REGS == 4096, "T=32, W=4 -> 4096 registers");
    CHECK(SIMTCore::PHYS_REGS == T * W * R, "PHYS_REGS");

    // 1. physical_addr formula + it is a bijection onto [0, PHYS_REGS)
    static bool seen[SIMTCore::PHYS_REGS];
    for (int w = 0; w < W; w++)
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++) {
                uint32_t p = SIMTCore::physical_addr(w, t, r);
                CHECK(p == (uint32_t)(w * T * 32 + t * 32 + r), "formula w=%d t=%d r=%d", w, t, r);
                CHECK(p < (uint32_t)SIMTCore::PHYS_REGS && !seen[p], "collision/out of range at %u", p);
                seen[p] = true;
            }

    // 2. Every (warp, thread, reg) holds its own value: write a unique value
    //    everywhere, read it all back (catches any bank/warp/reg aliasing).
    for (int w = 0; w < W; w++)
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++)
                core.write_reg(w, t, r, 0xA5000000u | SIMTCore::physical_addr(w, t, r));
    for (int w = 0; w < W; w++)
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++) {
                reg_t exp = r == 0 ? 0 : (0xA5000000u | SIMTCore::physical_addr(w, t, r));
                CHECK(core.read_reg(w, t, r) == exp, "readback w=%d t=%d r=%d got 0x%x", w, t, r, core.read_reg(w, t, r));
                // 3. physical-address view agrees with the (warp, thread, reg) view
                CHECK(core.read_phys(SIMTCore::physical_addr(w, t, r)) == exp, "read_phys w=%d t=%d r=%d", w, t, r);
            }

    // 4. x0 is zero for every thread, even after an explicit write
    for (int w = 0; w < W; w++)
        for (int t = 0; t < T; t++) {
            core.write_reg(w, t, 0, 0xFFFFFFFFu);
            CHECK(core.read_reg(w, t, 0) == 0, "x0 not hardwired (w=%d t=%d)", w, t);
        }

    // 5. Writing one thread's register doesn't disturb any other thread/warp
    SIMTCore c2(&mem);
    c2.write_reg(2, 3, 17, 0xDEADBEEF);
    for (int w = 0; w < W; w++)
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++) {
                bool target = (w == 2 && t == 3 && r == 17);
                CHECK(c2.read_reg(w, t, r) == (target ? 0xDEADBEEFu : 0u), "isolation w=%d t=%d r=%d", w, t, r);
            }

    std::printf(failures ? "REGFILE TEST FAILED (%d)\n" : "REGFILE TEST PASSED (%d failures)\n", failures);
    return failures ? 1 : 0;
}
