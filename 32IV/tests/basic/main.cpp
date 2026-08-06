#include <iostream>
#include <string>
#include <cstdlib>
#include "../../include/CPU.h"
#include "../../include/memory.h"
#include "loader.h"

int main(int argc, char* argv[]) {
    // Defaults preserve old behavior: `make run-test` with no args still works.
    std::string bin_path = "tests/basic/test.bin";
    int num_cycles = 500;

    if (argc > 1) bin_path = argv[1];
    if (argc > 2) num_cycles = std::atoi(argv[2]);

    std::cout << "Starting RISC-V CPU Emulator - Compiled C Program Test..." << std::endl;
    std::cout << "Loading: " << bin_path << " | Cycles: " << num_cycles << std::endl;

    Memory memory(1024 * 1024); // matches link.ld's LENGTH = 1M

    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load test binary. Exiting.\n";
        return 1;
    }

    CPU cpu(&memory);

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

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
