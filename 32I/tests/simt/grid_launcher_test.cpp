// Step B gate test for GridLauncher -- see docs/CUDA/grid_launch_plan.md.
// Hand-encoded RV32IM kernel (no compiler involved): every live lane records
// its hw_tid / sp / a0 / blockIdx / gridDim, so the test can recover which SM
// ran each block (hw_tid / (W*T)) and check stack placement against the
// host stack reserve passed in by the CPU side.
// Build/run:  make -C 32I run-grid-launcher-test
#include <cstdio>
#include <cstdint>
#include <vector>
#include "../../include/GridLauncher.h"
#include "../../include/memory.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---- tiny RV32 encoders (just what this kernel needs) ----
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

static const uint32_t OP_SIMT = SIMTCore::OPCODE_SIMT;
static uint32_t simt_tid(uint32_t rd)                 { return r_type(OP_SIMT, SIMTCore::SIMT_TID, 0, rd, 0, 0); }
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
static const uint32_t FIELD = 0x700;   // bytes per output field: room for 448 gids (<= 2047: fits a 12-bit imm)
static const uint32_t TPB   = 40;      // 2 warps per block, second one partially masked

// OUT + f*FIELD + gid*4, fields: 0 hw_tid, 1 sp, 2 a0, 3 blockIdx, 4 gridDim.
static std::vector<uint32_t> record_kernel() {
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
        simt_ident(SIMTCore::IDENT_HW_TID, 28),
        sw(9, 28, 0),
        sw(9, 2, FIELD),             // sp as seeded by the launcher
        addi(9, 9, FIELD),           // two steps: 2*FIELD overflows addi's 12-bit imm
        addi(9, 9, FIELD),
        sw(9, 10, 0),                // a0
        sw(9, 5, FIELD),             // blockIdx
        addi(9, 9, FIELD),           // two steps: 2*FIELD overflows addi's 12-bit imm
        addi(9, 9, FIELD),
        sw(9, 7, 0),                 // gridDim
        ret(),
    };
}

static uint32_t field(Memory& mem, uint32_t f, uint32_t gid) { return mem.read_word(OUT + f * FIELD + gid * 4); }

static void load_code(Memory& mem, const std::vector<uint32_t>& code) {
    for (size_t i = 0; i < code.size(); i++) mem.write_word(ENTRY + 4 * i, code[i]);
}

// Launch num_blocks blocks and check block->SM map, stacks, args and cycle accounting.
static void check_launch(uint32_t num_blocks, size_t mem_bytes, uint32_t host_reserve) {
    Memory mem(mem_bytes);
    std::vector<uint32_t> code = record_kernel();
    load_code(mem, code);

    GridLauncher gpu(&mem, host_reserve);
    const uint32_t args = 0xA5000000u | num_blocks;
    uint64_t cycles = gpu.launch_grid(ENTRY, TPB, num_blocks, args);

    const uint32_t stack_top = (uint32_t)mem_bytes - host_reserve;
    CHECK(gpu.stack_top() == stack_top, "[%u blocks] stack_top %x, want %x", num_blocks, gpu.stack_top(), stack_top);

    const uint32_t per_sm_threads = SIMTCore::WARPS_RESIDENT * SIMTCore::THREADS_PER_WARP;
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t want_sm = b % NUM_SMS; // round-robin
        for (uint32_t t = 0; t < TPB; t++) {
            uint32_t gid = b * TPB + t;
            uint32_t hw = field(mem, 0, gid);
            CHECK(hw / per_sm_threads == want_sm, "[%u blocks] block %u thr %u ran on SM %u, want %u",
                  num_blocks, b, t, hw / per_sm_threads, want_sm);
            CHECK(hw == want_sm * per_sm_threads + t, "[%u blocks] block %u thr %u hw_tid %u", num_blocks, b, t, hw);
            uint32_t want_sp = stack_top - (hw + 1) * SIMTCore::STACK_BYTES_PER_THREAD;
            CHECK(field(mem, 1, gid) == want_sp, "[%u blocks] block %u thr %u sp %x, want %x",
                  num_blocks, b, t, field(mem, 1, gid), want_sp);
            CHECK(field(mem, 2, gid) == args, "[%u blocks] block %u thr %u a0", num_blocks, b, t);
            CHECK(field(mem, 3, gid) == b, "[%u blocks] block %u thr %u blockIdx %u", num_blocks, b, t, field(mem, 3, gid));
            CHECK(field(mem, 4, gid) == num_blocks, "[%u blocks] block %u thr %u gridDim %u", num_blocks, b, t, field(mem, 4, gid));
        }
    }
    // No thread beyond the grid wrote anything.
    for (uint32_t gid = num_blocks * TPB; gid < num_blocks * TPB + TPB; gid++)
        CHECK(field(mem, 3, gid) == 0 && field(mem, 2, gid) == 0, "[%u blocks] stray write @gid %u", num_blocks, gid);

    // Cycles: every block issues 2 warps x kernel length; SMs run in parallel,
    // blocks on one SM back to back, so the launch costs the busiest SM's total.
    const uint64_t per_block = 2 * code.size();
    uint64_t want_max = 0;
    for (uint32_t s = 0; s < NUM_SMS; s++) {
        uint64_t nb_on_sm = num_blocks / NUM_SMS + (s < num_blocks % NUM_SMS ? 1 : 0);
        uint64_t want = nb_on_sm * per_block;
        CHECK(gpu.sm_cycles(s) == want, "[%u blocks] SM %u cycles %llu, want %llu", num_blocks, s,
              (unsigned long long)gpu.sm_cycles(s), (unsigned long long)want);
        if (want > want_max) want_max = want;
    }
    CHECK(cycles == want_max, "[%u blocks] launch returned %llu, want %llu", num_blocks,
          (unsigned long long)cycles, (unsigned long long)want_max);
}

static void test_launch_wrapping_block_counts() {
    const uint32_t counts[] = {1, NUM_SMS - 1, NUM_SMS, NUM_SMS + 1, 2 * NUM_SMS + 1};
    for (uint32_t nb : counts) check_launch(nb, 4 * 1024 * 1024, CPU::HOST_STACK_RESERVE);
}

// The reserve is whatever the CPU side hands over, and stack_top follows the RAM size.
static void test_reserve_and_ram_size_are_honoured() {
    check_launch(NUM_SMS + 1, 2 * 1024 * 1024, 0x8000);
}

static void test_invalid_launches_rejected() {
    Memory mem(4 * 1024 * 1024);
    load_code(mem, record_kernel());
    GridLauncher gpu(&mem, CPU::HOST_STACK_RESERVE);
    struct { uint32_t tpb, nb; const char* what; } bad[] = {
        {0, 1, "threads_per_block 0"},
        {SIMTCore::MAX_THREADS_PER_BLOCK + 1, 1, "threads_per_block > max"},
        {TPB, 0, "num_blocks 0"},
        {TPB, MAX_BLOCKS + 1, "num_blocks > MAX_BLOCKS"},
    };
    for (auto& c : bad) {
        CHECK(gpu.launch_grid(ENTRY, c.tpb, c.nb, 0) == 0, "%s should return 0 cycles", c.what);
        for (uint32_t s = 0; s < NUM_SMS; s++)
            CHECK(gpu.sm_cycles(s) == 0, "%s: SM %u ran", c.what, s);
        CHECK(field(mem, 4, 0) == 0, "%s: a kernel wrote output", c.what);
    }
}

// RAM that can't hold the host reserve plus every SM's device stacks must be
// refused up front, not handed out as sp values outside RAM.
static void test_ram_too_small_for_stacks_rejected() {
    const uint32_t device_stacks = NUM_SMS * SIMTCore::MAX_THREADS_PER_BLOCK * SIMTCore::STACK_BYTES_PER_THREAD;
    struct { size_t ram; uint32_t reserve; const char* what; } bad[] = {
        {256 * 1024, 512 * 1024, "reserve > RAM"},
        {CPU::HOST_STACK_RESERVE + device_stacks - 4, CPU::HOST_STACK_RESERVE, "RAM 4 bytes short of reserve + device stacks"},
    };
    for (auto& c : bad) {
        Memory mem(c.ram);
        load_code(mem, record_kernel());
        GridLauncher gpu(&mem, c.reserve);
        CHECK(gpu.launch_grid(ENTRY, TPB, 1, 0) == 0, "%s: launch should be refused", c.what);
        CHECK(gpu.sm_cycles(0) == 0, "%s: SM 0 ran", c.what);
    }
    // Exactly enough RAM for the reserve and the stacks is fine (code + OUT sit at 0x1000/0x100000).
    Memory mem(0x180000);
    load_code(mem, record_kernel());
    GridLauncher gpu(&mem, 0x180000 - device_stacks);
    CHECK(gpu.launch_grid(ENTRY, TPB, 1, 0) != 0, "RAM exactly reserve + device stacks should launch");
}

// A block that hits the issue cap abandons the rest of the grid instead of
// burning the cap again on every remaining block, and says so.
static void test_runaway_kernel_stops_grid() {
    Memory mem(4 * 1024 * 1024);
    // nop ; j -4  -- a two-instruction loop (a lone self-jump would be the halt idiom).
    load_code(mem, {addi(0, 0, 0), 0xFFDFF06Fu /* jal x0, -4 */});
    GridLauncher gpu(&mem, CPU::HOST_STACK_RESERVE);
    gpu.set_max_issues_per_block(500);
    gpu.launch_grid(ENTRY, 32, 8, 0);
    CHECK(gpu.timed_out(), "runaway grid should report timed_out");
    CHECK(gpu.sm_cycles(0) == 500, "SM 0 cycles %llu, want exactly the cap", (unsigned long long)gpu.sm_cycles(0));
    for (uint32_t s = 1; s < NUM_SMS; s++)
        CHECK(gpu.sm_cycles(s) == 0, "SM %u ran a block after the grid was abandoned", s);

    load_code(mem, record_kernel());
    gpu.launch_grid(ENTRY, TPB, 1, 0);
    CHECK(!gpu.timed_out(), "a clean launch should clear timed_out");
}

int main() {
    test_launch_wrapping_block_counts();
    test_reserve_and_ram_size_are_honoured();
    test_invalid_launches_rejected();
    test_ram_too_small_for_stacks_rejected();
    test_runaway_kernel_stops_grid();
    if (failures == 0) std::printf("ALL GridLauncher TESTS PASSED\n");
    else std::printf("%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
