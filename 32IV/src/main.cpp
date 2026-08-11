#include <iostream>
#include <cstdint>
#include "../include/CPU.h"
#include "../include/memory.h"

// ---------------------------------------------------------------------------
// Tiny instruction encoders (built by shifting fields, not hand-computed hex)
// so the hand-assembled test program below is easy to read and hard to get
// wrong. Only the subset of RV32I/V used by this test is covered.
// ---------------------------------------------------------------------------
static uint32_t enc_addi(uint32_t rd, uint32_t rs1, int32_t imm) {
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0x0 << 12) | ((rd & 0x1F) << 7) | 0x13;
}

static uint32_t enc_lw(uint32_t rd, uint32_t rs1, int32_t imm) {
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | (0x2 << 12) | ((rd & 0x1F) << 7) | 0x03;
}

static uint32_t enc_beq(uint32_t rs1, uint32_t rs2, int32_t imm) {
    // Only used here for imm=0 (infinite-loop halt), so the full B-type immediate
    // scramble is unnecessary - keep it simple and explicit for that one case.
    return ((imm & 0x1) << 7) | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20) | (0x0 << 12) | 0x63;
}

// vsetvli rd, rs1, e<8*2^sew>, m<lmul>  (LMUL encodings: 0=m1,1=m2,2=m4,3=m8)
static uint32_t enc_vsetvli(uint32_t rd, uint32_t rs1, uint32_t sew, uint32_t lmul) {
    uint32_t zimm = ((sew & 0x7) << 3) | (lmul & 0x7);
    return (zimm << 20) | ((rs1 & 0x1F) << 15) | (0x7 << 12) | ((rd & 0x1F) << 7) | 0x57;
}

// vle<eew>.v vd, (rs1) - unit-stride, unmasked (vm=1) or masked via v0 (vm=0)
// width: 0=8-bit, 5=16-bit, 6=32-bit, 7=64-bit
static uint32_t enc_vload(uint32_t vd, uint32_t rs1, uint32_t width, bool unmasked) {
    return ((unmasked ? 1u : 0u) << 25) | ((rs1 & 0x1F) << 15) | ((width & 0x7) << 12) | ((vd & 0x1F) << 7) | 0x07;
}

// vse<eew>.v vs3, (rs1) - unit-stride, unmasked or masked via v0
static uint32_t enc_vstore(uint32_t vs3, uint32_t rs1, uint32_t width, bool unmasked) {
    return ((unmasked ? 1u : 0u) << 25) | ((rs1 & 0x1F) << 15) | ((width & 0x7) << 12) | ((vs3 & 0x1F) << 7) | 0x27;
}

int main() {
    std::cout << "Starting RISC-V CPU Emulator - Vector Load/Store Verification..." << std::endl;

    Memory memory(1024 * 1024);

    // -----------------------------------------------------------------
    // Seed data memory before the program runs
    // -----------------------------------------------------------------
    const uint32_t SRC_ADDR      = 0x200; // 4 words: 0x11111111, 0x22222222, 0x33333333, 0x44444444
    const uint32_t ROUNDTRIP_DST = 0x240; // unmasked vle32.v/vse32.v round-trip target
    const uint32_t SENTINEL_ADDR = 0x280; // 4 words of 0xEEEEEEEE, to seed v2 before a masked load
    const uint32_t MASK_ADDR     = 0x2C0; // 1 byte: mask pattern 0b0101 (lanes 0,2 active)
    const uint32_t MASKED_LD_DST = 0x2E0; // masked-load result, stored back out for inspection
    const uint32_t STORE_SENTINEL_ADDR = 0x320; // 4 words of 0xCCCCCCCC, to verify masked store
                                                 // leaves inactive-lane bytes untouched

    memory.write_word(SRC_ADDR + 0,  0x11111111);
    memory.write_word(SRC_ADDR + 4,  0x22222222);
    memory.write_word(SRC_ADDR + 8,  0x33333333);
    memory.write_word(SRC_ADDR + 12, 0x44444444);

    memory.write_word(SENTINEL_ADDR + 0,  0xEEEEEEEE);
    memory.write_word(SENTINEL_ADDR + 4,  0xEEEEEEEE);
    memory.write_word(SENTINEL_ADDR + 8,  0xEEEEEEEE);
    memory.write_word(SENTINEL_ADDR + 12, 0xEEEEEEEE);

    memory.write_byte(MASK_ADDR, 0x05); // 0b00000101 -> lanes 0 and 2 active, 1 and 3 masked out

    memory.write_word(STORE_SENTINEL_ADDR + 0,  0xCCCCCCCC);
    memory.write_word(STORE_SENTINEL_ADDR + 4,  0xCCCCCCCC);
    memory.write_word(STORE_SENTINEL_ADDR + 8,  0xCCCCCCCC);
    memory.write_word(STORE_SENTINEL_ADDR + 12, 0xCCCCCCCC);

    // -----------------------------------------------------------------
    // Program (assembled sequentially starting at address 0)
    // -----------------------------------------------------------------
    uint32_t addr = 0x00;
    auto emit = [&](uint32_t instr) { memory.write_word(addr, instr); addr += 4; };

    // --- Part 1: unmasked round trip, unit-stride, EEW=32, vl=4 ---
    emit(enc_addi(10, 0, 4));                 // x10 = 4  (AVL)
    emit(enc_vsetvli(11, 10, /*sew=32*/2, /*lmul=1*/0)); // x11 = vl (expect 4)
    emit(enc_addi(5, 0, SRC_ADDR));            // x5 = source base address
    emit(enc_vload(1, 5, /*width=32*/6, /*unmasked*/true));  // vle32.v v1, (x5)
    emit(enc_addi(6, 0, ROUNDTRIP_DST));       // x6 = round-trip destination
    emit(enc_vstore(1, 6, /*width=32*/6, /*unmasked*/true)); // vse32.v v1, (x6)
    emit(enc_lw(20, 6, 0));
    emit(enc_lw(21, 6, 4));
    emit(enc_lw(22, 6, 8));
    emit(enc_lw(23, 6, 12));                   // x20..x23 should read back 0x11111111.. 0x44444444

    // --- Part 2: seed v2 with a sentinel, then load the mask into v0 ---
    emit(enc_addi(7, 0, SENTINEL_ADDR));
    emit(enc_vload(2, 7, /*width=32*/6, /*unmasked*/true));  // v2 = 0xEEEEEEEE x4 (sentinel)
    emit(enc_addi(8, 0, 1));                   // x8 = 1 (AVL for the e8 mask load)
    emit(enc_vsetvli(14, 8, /*sew=8*/0, /*lmul=1*/0));        // vl=1, sew=8 (touches only v0 byte 0)
    emit(enc_addi(9, 0, MASK_ADDR));
    emit(enc_vload(0, 9, /*width=8*/0, /*unmasked*/true));    // vle8.v v0, (x9) - loads the mask

    // --- Part 3: masked load using v0, lanes 0 and 2 active ---
    emit(enc_vsetvli(11, 10, /*sew=32*/2, /*lmul=1*/0));      // reconfigure back to vl=4, sew=32
    emit(enc_vload(2, 5, /*width=32*/6, /*unmasked*/false));  // vle32.v v2, (x5), v0.t
    emit(enc_addi(16, 0, MASKED_LD_DST));
    emit(enc_vstore(2, 16, /*width=32*/6, /*unmasked*/true)); // store v2 back out for inspection
    emit(enc_lw(24, 16, 0));
    emit(enc_lw(25, 16, 4));
    emit(enc_lw(26, 16, 8));
    emit(enc_lw(27, 16, 12));
    // Expect: x24=0x11111111 (lane0 active), x25=0xEEEEEEEE (lane1 kept sentinel),
    //         x26=0x33333333 (lane2 active), x27=0xEEEEEEEE (lane3 kept sentinel)

    // --- Part 4: masked store using v0, verify inactive lanes leave memory untouched ---
    emit(enc_addi(17, 0, STORE_SENTINEL_ADDR));
    emit(enc_vstore(1, 17, /*width=32*/6, /*unmasked*/false)); // vse32.v v1, (x17), v0.t
    emit(enc_lw(28, 17, 0));
    emit(enc_lw(29, 17, 4));
    emit(enc_lw(30, 17, 8));
    emit(enc_lw(31, 17, 12));
    // Expect: x28=0x11111111 (lane0 written), x29=0xCCCCCCCC (lane1 untouched),
    //         x30=0x33333333 (lane2 written), x31=0xCCCCCCCC (lane3 untouched)

    emit(enc_beq(0, 0, 0)); // Halt / infinite loop at end

    CPU cpu(&memory);

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    int num_cycles = 40;
    for (int i = 0; i < num_cycles; i++) {
        cpu.fetch();
        cpu.decode();
        cpu.execute();
        cpu.read();
        cpu.writeback();
    }

    std::cout << "\n--- Final State ---" << std::endl;
    cpu.print_state();

    std::cout << "\n--- Vector Registers ---" << std::endl;
    cpu.print_vector_reg(0); // mask register
    cpu.print_vector_reg(1); // unmasked round-trip source
    cpu.print_vector_reg(2); // masked-load result

    std::cout << "\n--- Verification Checklist ---" << std::endl;
    std::cout << "1. Round trip: x20..x23 == 0x11111111, 0x22222222, 0x33333333, 0x44444444?" << std::endl;
    std::cout << "2. Masked load: x24=0x11111111, x25=0xEEEEEEEE, x26=0x33333333, x27=0xEEEEEEEE?" << std::endl;
    std::cout << "3. Masked store: x28=0x11111111, x29=0xCCCCCCCC, x30=0x33333333, x31=0xCCCCCCCC?" << std::endl;

    return 0;
}
