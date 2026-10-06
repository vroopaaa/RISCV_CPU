// Grid-launch harness for tests/simt/run_grid_tests.py: loads an assembled
// kernel binary, launches it over the GPU through GridLauncher::launch_grid
// (directly -- the host CPU's LAUNCH opcode is phase C), prints the launch
// result on one line and dumps a memory region for the Python driver to diff.
//
// Usage: grid_harness <bin_path> <entry_hex> <threads_per_block> <num_blocks>
//                     <args_hex> <max_issues_per_block> <dump_base_hex>
//                     <dump_size_hex> <dump_file_path>
// Prints: cycles=<n> timed_out=<0|1> sm_cycles=<sm0>,<sm1>,...
#include <iostream>
#include <string>
#include <cstdlib>
#include "../../include/GridLauncher.h"
#include "../../include/memory.h"
#include "../basic/loader.h"

int main(int argc, char* argv[]) {
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " <bin_path> <entry_hex> <threads_per_block> <num_blocks>"
                  << " <args_hex> <max_issues_per_block> <dump_base_hex> <dump_size_hex> <dump_file_path>\n";
        return 1;
    }
    std::string bin_path       = argv[1];
    uint32_t entry             = std::strtoul(argv[2], nullptr, 16);
    uint32_t threads_per_block = std::strtoul(argv[3], nullptr, 10);
    uint32_t num_blocks        = std::strtoul(argv[4], nullptr, 10);
    uint32_t args              = std::strtoul(argv[5], nullptr, 16);
    uint64_t max_issues        = std::strtoull(argv[6], nullptr, 10);
    uint32_t dump_base         = std::strtoul(argv[7], nullptr, 16);
    uint32_t dump_size         = std::strtoul(argv[8], nullptr, 16);
    std::string dump_path      = argv[9];

    // Same 4MB RAM as tests/python/link.ld; the host keeps the top of it for its stack.
    Memory memory(4 * 1024 * 1024);
    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load " << bin_path << "\n";
        return 1;
    }

    GridLauncher gpu(&memory, CPU::HOST_STACK_RESERVE);
    gpu.set_max_issues_per_block(max_issues);
    uint64_t cycles = gpu.launch_grid(entry, threads_per_block, num_blocks, args);

    std::cout << "cycles=" << cycles << " timed_out=" << (gpu.timed_out() ? 1 : 0) << " sm_cycles=";
    for (uint32_t sm = 0; sm < NUM_SMS; sm++) std::cout << (sm ? "," : "") << gpu.sm_cycles((int)sm);
    std::cout << "\n";

    memory.dump_range(dump_base, dump_base + dump_size, dump_path);
    return 0;
}
