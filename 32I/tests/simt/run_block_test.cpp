// Step A gate test for SIMTCore::run_block() / the SIMT_IDENT ops -- see
// docs/CUDA/grid_launch_plan.md. Kernels are hand-encoded RV32IM words (no
// compiler involved), so this checks the SM core in isolation before the
// grid launcher / host CPU / C macros exist.
// Build/run:  make -C 32I run-simt-runblock-test
#include <cstdio>
#include <cstdint>
#include <vector>
#include "../../include/SIMT.h"
#include "../../include/memory.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---- tiny RV32 encoders (just what these kernels need) ----
static uint32_t r_type(uint32_t op, uint32_t f3, uint32_t f7, uint32_t rd, uint32_t rs1, uint32_t rs2) {
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t i_type(uint32_t op, uint32_t f3, uint32_t rd, uint32_t rs1, int32_t imm) {
    return ((uint32_t)(imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t s_type(uint32_t f3, uint32_t rs1, uint32_t rs2, int32_t imm) {
    return (((uint32_t)(imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           (((uint32_t)imm & 0x1F) << 7) | 0x23;
}
static uint32_t u_type(uint32_t op, uint32_t rd, uint32_t imm20) { return (imm20 << 12) | (rd << 7) | op; }
static uint32_t jal(uint32_t rd, int32_t off) {
    uint32_t u = (uint32_t)off;
    return (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3FF) << 21) | (((u >> 11) & 1) << 20) |
           (((u >> 12) & 0xFF) << 12) | (rd << 7) | 0x6F;
}

static const uint32_t OP_SIMT = SIMTCore::OPCODE_SIMT;
static uint32_t simt_tid(uint32_t rd)               { return r_type(OP_SIMT, SIMTCore::SIMT_TID, 0, rd, 0, 0); }
static uint32_t simt_ident(uint32_t sel, uint32_t rd) { return r_type(OP_SIMT, SIMTCore::SIMT_IDENT, sel, rd, 0, 0); }
static uint32_t addi(uint32_t rd, uint32_t rs1, int32_t imm) { return i_type(0x13, 0, rd, rs1, imm); }
static uint32_t srli(uint32_t rd, uint32_t rs1, int32_t sh)  { return i_type(0x13, 5, rd, rs1, sh); }
static uint32_t andi(uint32_t rd, uint32_t rs1, int32_t imm) { return i_type(0x13, 7, rd, rs1, imm); }
static uint32_t slli(uint32_t rd, uint32_t rs1, int32_t sh)  { return i_type(0x13, 1, rd, rs1, sh); }
static uint32_t add(uint32_t rd, uint32_t rs1, uint32_t rs2) { return r_type(0x33, 0, 0, rd, rs1, rs2); }
static uint32_t mul(uint32_t rd, uint32_t rs1, uint32_t rs2) { return r_type(0x33, 0, 1, rd, rs1, rs2); }
static uint32_t sw(uint32_t rs1, uint32_t rs2, int32_t imm)  { return s_type(2, rs1, rs2, imm); }
static uint32_t ret()                                        { return i_type(0x67, 0, 0, 1, 0); } // jalr x0, ra, 0

static const uint32_t ENTRY = 0x1000;
static const uint32_t OUT   = 0x100000;
static const uint32_t STACK_TOP = 0x3F0000;

static void load_code(Memory& mem, const std::vector<uint32_t>& code) {
    for (size_t i = 0; i < code.size(); i++) mem.write_word(ENTRY + 4 * i, code[i]);
}

// Common prologue: x9 = &OUT[gid] where gid = blockIdx*blockDim + threadIdx.
// Leaves x5 = blockIdx, x6 = blockDim, x7 = gridDim, x30 = threadIdx.
static std::vector<uint32_t> identity_prologue() {
    return {
        simt_ident(SIMTCore::IDENT_BLOCK_IDX, 5),
        simt_ident(SIMTCore::IDENT_BLOCK_DIM, 6),
        simt_ident(SIMTCore::IDENT_GRID_DIM, 7),
        simt_tid(29),
        srli(30, 29, 16),            // threadIdx = TID[23:16]
        andi(30, 30, 0xFF),
        mul(31, 5, 6),
        add(31, 31, 30),             // gid
        slli(31, 31, 2),
        u_type(0x37, 9, OUT >> 12),  // lui x9, OUT
        add(9, 9, 31),
    };
}

static void test_identity_and_partial_warp() {
    Memory mem(4 * 1024 * 1024);
    std::vector<uint32_t> code = identity_prologue();
    code.push_back(sw(9, 5, 0x000));   // field 0: blockIdx
    code.push_back(sw(9, 6, 0x200));   // field 1: blockDim
    code.push_back(sw(9, 7, 0x400));   // field 2: gridDim
    code.push_back(sw(9, 30, 0x600));  // field 3: threadIdx
    code.push_back(ret());
    load_code(mem, code);

    SIMTCore core(&mem);
    SIMTCore::BlockLaunch b{ENTRY, /*block_id=*/2, /*block_dim=*/40, /*grid_dim=*/3, /*args=*/0, STACK_TOP};
    bool timed_out = true;
    uint64_t issued = core.run_block(b, &timed_out);

    CHECK(!timed_out, "identity kernel timed out");
    // 2 warps: warp 0 (32 lanes) + warp 1 (8 live lanes) each run the same instruction stream.
    CHECK(issued == 2 * code.size(), "issued %llu, expected %zu", (unsigned long long)issued, 2 * code.size());
    CHECK(core.warp_halted(0) && core.warp_halted(1), "both warps should have halted via the ret sentinel");
    CHECK(core.warp_pc(0) == SIMTCore::BLOCK_RETURN_PC, "warp 0 pc != BLOCK_RETURN_PC");

    for (uint32_t i = 0; i < 40; i++) {
        uint32_t gid = 2 * 40 + i;
        CHECK(mem.read_word(OUT + 0x000 + gid * 4) == 2,  "blockIdx  @gid %u", gid);
        CHECK(mem.read_word(OUT + 0x200 + gid * 4) == 40, "blockDim  @gid %u", gid);
        CHECK(mem.read_word(OUT + 0x400 + gid * 4) == 3,  "gridDim   @gid %u", gid);
        CHECK(mem.read_word(OUT + 0x600 + gid * 4) == i,  "threadIdx @gid %u", gid);
    }
    // Lanes 8..31 of warp 1 (threadIdx 40..63 -> gid 120..143) were masked off: must not have written.
    for (uint32_t gid = 120; gid < 144; gid++)
        for (uint32_t f = 0; f < 4; f++)
            CHECK(mem.read_word(OUT + f * 0x200 + gid * 4) == 0, "masked lane wrote field %u @gid %u", f, gid);
    // Nothing below this block's slice was touched either.
    for (uint32_t gid = 0; gid < 80; gid++)
        CHECK(mem.read_word(OUT + gid * 4) == 0, "stray write @gid %u", gid);
}

static void test_stack_args_hwtid_on_sm1() {
    Memory mem(4 * 1024 * 1024);
    std::vector<uint32_t> code = identity_prologue();
    code.push_back(simt_ident(SIMTCore::IDENT_HW_TID, 28));
    code.push_back(sw(9, 28, 0x000));  // hw_tid
    code.push_back(sw(9, 2,  0x200));  // sp as seeded by the launcher
    code.push_back(sw(9, 10, 0x400));  // a0 (kernel args)
    code.push_back(ret());
    load_code(mem, code);

    const int SM = 1;
    SIMTCore core(&mem, SM);
    SIMTCore::BlockLaunch b{ENTRY, 0, 64, 1, /*args=*/0xCAFE0000u, STACK_TOP};
    bool timed_out = true;
    core.run_block(b, &timed_out);
    CHECK(!timed_out, "stack kernel timed out");

    for (uint32_t thr = 0; thr < 64; thr++) {
        uint32_t expect_hw = (SM * SIMTCore::WARPS_RESIDENT) * SIMTCore::THREADS_PER_WARP + thr; // slot = thr/32, lane = thr%32
        CHECK(mem.read_word(OUT + thr * 4) == expect_hw, "hw_tid thr %u: got %u want %u", thr,
              mem.read_word(OUT + thr * 4), expect_hw);
        uint32_t expect_sp = STACK_TOP - (expect_hw + 1) * SIMTCore::STACK_BYTES_PER_THREAD;
        CHECK(mem.read_word(OUT + 0x200 + thr * 4) == expect_sp, "sp thr %u: got %x want %x", thr,
              mem.read_word(OUT + 0x200 + thr * 4), expect_sp);
        CHECK(mem.read_word(OUT + 0x400 + thr * 4) == 0xCAFE0000u, "a0 thr %u", thr);
    }
}

static void test_timeout() {
    Memory mem(4 * 1024 * 1024);
    // nop ; j -4  -- a two-instruction loop (a lone self-jump would be the halt idiom).
    load_code(mem, {addi(0, 0, 0), jal(0, -4)});
    SIMTCore core(&mem);
    SIMTCore::BlockLaunch b{ENTRY, 0, 32, 1, 0, STACK_TOP};
    bool timed_out = false;
    uint64_t issued = core.run_block(b, &timed_out, 1000);
    CHECK(timed_out, "runaway kernel should report timed_out");
    CHECK(issued == 1000, "issued %llu, expected exactly the cap", (unsigned long long)issued);
}

static void test_bad_block_dim() {
    Memory mem(4 * 1024 * 1024);
    load_code(mem, {ret()});
    SIMTCore core(&mem);
    SIMTCore::BlockLaunch b{ENTRY, 0, 0, 1, 0, STACK_TOP};
    CHECK(core.run_block(b) == 0, "block_dim 0 should be rejected");
    b.block_dim = SIMTCore::MAX_THREADS_PER_BLOCK + 1;
    CHECK(core.run_block(b) == 0, "block_dim 129 should be rejected");
    b.block_dim = SIMTCore::MAX_THREADS_PER_BLOCK; // exactly the limit is fine
    CHECK(core.run_block(b) == 4 * 1, "128 threads = 4 warps x 1 instruction (ret)");
}

int main() {
    test_identity_and_partial_warp();
    test_stack_args_hwtid_on_sm1();
    test_timeout();
    test_bad_block_dim();
    if (failures == 0) std::printf("ALL run_block TESTS PASSED\n");
    else std::printf("%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
