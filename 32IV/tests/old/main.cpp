#include <iostream>
#include "../../include/CPU.h"
#include "../../include/memory.h"
#include "loader.h"

int main() {
    std::cout << "Starting RISC-V CPU Emulator - Compiled C Program Test..." << std::endl;

    Memory memory(1024 * 1024); // 1MB, matches link.ld's LENGTH = 1M

    // Load the compiled RISC-V binary instead of hand-writing instructions
    if (!load_binary(memory, "tests/basic/test.bin", 0x0)) {
        std::cerr << "Failed to load test binary. Exiting.\n";
        return 1;
    }

    CPU cpu(&memory);

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    // Compiled programs are less predictable in length than hand-written tests,
    // so give it more headroom. You can tune this once you know roughly how
    // many instructions your test program disassembles to.
    int num_cycles = 500;
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