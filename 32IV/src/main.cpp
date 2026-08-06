#include <iostream>
#include "../include/CPU.h"
#include "../include/memory.h" 

int main() {
    std::cout << "Starting RISC-V CPU Emulator - Vector Configuration Test (vsetvli)..." << std::endl;

    Memory memory(1024*1024); // 1KB of memory for testing

    // =========================================================
    // PROGRAM: vsetvli Verification Suite
    // DESCRIPTION: Tests normal AVL negotiation, maximum VL 
    // requesting (rs1=x0), and illegal state trapping (vill).
    // Note: The exact vl values in rd will depend on your emulator's VLEN.
    // =========================================================

    // --- TEST 1: Standard AVL Negotiation ---
    // We want to process 100 elements. We request e32 (32-bit elements) and m1 (LMUL=1).
    // vtypei = SEW(010) LMUL(000) = 0x10.
    memory.write_word(0x00, 0x06400513); // ADDI x10, x0, 100       (a0 = 100) -> Set requested AVL
    memory.write_word(0x04, 0x010575d7); // VSETVLI x11, x10, e32, m1
                                         // EXPECT: x11 (a1) gets the negotiated vl. 
                                         // If VLEN=128, VLMAX=4. Since 100 > 4, x11 gets 4.

    // --- TEST 2: Maximum Vector Length (rs1 = x0) ---
    // Passing x0 as the source register means "give me the maximum VL possible".
    // We request e16 (16-bit elements) and m2 (LMUL=2).
    // vtypei = SEW(001) LMUL(001) = 0x09.
    memory.write_word(0x08, 0x00907657); // VSETVLI x12, x0, e16, m2
                                         // EXPECT: x12 (a2) gets VLMAX. 
                                         // If VLEN=128, VLMAX = (128/16) * 2 = 16. x12 gets 16.

    // --- TEST 3: Illegal Configuration (vill Trap) ---
    // We intentionally request an illegal SEW size (e.g., SEW=100 in binary, which is reserved).
    // vtypei = SEW(100) LMUL(000) = 0x20.
    memory.write_word(0x0C, 0x020576d7); // VSETVLI x13, x10, 0x20
                                         // EXPECT: x13 (a3) gets 0. The internal vtype CSR MSB (vill) is set to 1.
                                         // Internal vl CSR is set to 0.

    // --- END PROGRAM ---
    memory.write_word(0x10, 0x00000063); // BEQ x0, x0, 0         (Halt / Infinite Loop at End)
    memory.write_word(0x14, 0x00000013); // NOP


    CPU cpu(&memory);

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    // 20 cycles is more than enough to run this small configuration test
    int num_cycles = 20;
    for (int i = 0; i < num_cycles; i++) {
        cpu.fetch();
        cpu.decode();
        
        // Ensure your router successfully routes OP-V (0x57) to vector_decode 
        // and vector_config_execute based on the fields we discussed.
        cpu.execute(); 
        
        cpu.read();
        cpu.writeback();
    }

    std::cout << "\n--- Final State ---" << std::endl;
    cpu.print_state();

    // Verification check for the user reading the terminal:
    std::cout << "\n--- Verification Checklist ---" << std::endl;
    std::cout << "1. Does x11 contain the clamped AVL? (Test 1)" << std::endl;
    std::cout << "2. Does x12 contain the exact VLMAX? (Test 2)" << std::endl;
    std::cout << "3. Is x13 strictly 0 due to the vill trap? (Test 3)" << std::endl;

    return 0;
}