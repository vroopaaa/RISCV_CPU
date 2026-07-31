#include <iostream>
#include "../include/CPU.h"
#include "../include/memory.h" 

int main() {
    std::cout << "Starting RISC-V CPU Emulator - Hardware MUL Factorial Test..." << std::endl;

    Memory memory(1024*1024); // 1KB of memory for testing

    // =========================================================
    // PROGRAM: Recursive Factorial (5!) using M-Extension
    // EXPECTED OUTPUT: x10 (a0) should equal 0x00000078 (120)
    // =========================================================

    // --- MAIN SETUP ---
    memory.write_word(0x00, 0x10000113); // ADDI x2, x0, 256     (sp = 256) -> Initialize Stack
    memory.write_word(0x04, 0x00500513); // ADDI x10, x0, 5      (a0 = 5)   -> Set argument n = 5
    memory.write_word(0x08, 0x00c000ef); // JAL x1, 12           (call fact at 0x14)
    memory.write_word(0x0C, 0x00000063); // BEQ x0, x0, 0        (Halt / Infinite Loop at End)
    memory.write_word(0x10, 0x00000013); // NOP

    // --- FACTORIAL FUNCTION ---
    memory.write_word(0x14, 0xff410113); // ADDI x2, x2, -12     (sp -= 12) -> Allocate stack space
    memory.write_word(0x18, 0x00112423); // SW x1, 8(x2)         (save ra)
    memory.write_word(0x1C, 0x00a12223); // SW x10, 4(x2)        (save original n)

    // --- Base Case Check ---
    memory.write_word(0x20, 0x00100293); // ADDI x5, x0, 1       (t0 = 1)
    memory.write_word(0x24, 0x00a2c663); // BLT x5, x10, 12      (if 1 < n, goto Recursive Step at 0x30)

    // --- Return 1 (Base Case) ---
    memory.write_word(0x28, 0x00100513); // ADDI x10, x0, 1      (a0 = 1)
    memory.write_word(0x2C, 0x0140006f); // JAL x0, 20           (goto Epilogue at 0x40)

    // --- Recursive Step ---
    memory.write_word(0x30, 0xfff50513); // ADDI x10, x10, -1    (a0 = n - 1)
    memory.write_word(0x34, 0xfe1ff0ef); // JAL x1, -32          (recursive call fact at 0x14)

    // --- HARDWARE MULTIPLY (The M-Extension Magic!) ---
    memory.write_word(0x38, 0x00412583); // LW x11, 4(x2)        (a1 = original n from stack)
    memory.write_word(0x3C, 0x02a58533); // MUL x10, x11, x10    (a0 = a1 * a0) -> Hardware math!

    // --- Epilogue ---
    memory.write_word(0x40, 0x00812083); // LW x1, 8(x2)         (restore ra)
    memory.write_word(0x44, 0x00c10113); // ADDI x2, x2, 12      (sp += 12) -> Free stack space
    memory.write_word(0x48, 0x00008067); // JALR x0, 0(x1)       (return to caller)


    CPU cpu(&memory);

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    // Because we removed the software multiplier loop, the program is WAY faster!
    // We only need about 100 cycles now instead of 250.
    int num_cycles = 100;
    for (int i = 0; i < num_cycles; i++) {
        cpu.fetch();
        cpu.decode();
        cpu.execute();
        cpu.read();
        cpu.writeback();
    }

    std::cout << "\n--- Final State ---" << std::endl;
    cpu.print_state();

    return 0;
}