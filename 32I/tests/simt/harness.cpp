// Shared harness for every SIMT C-kernel test -- the SIMT-layer equivalent
// of tests/basic/main.cpp / tests/npu's build/run_test: loads a
// cross-compiled bare-metal binary, runs it (here: through SIMTCore's
// single-warp lane-broadcast engine instead of the scalar CPU), and dumps a
// memory region to a file for a Python driver to diff against an
// independently-computed expected result. See tests/simt/run_simt_tests.py.
//
// Usage: simt_harness <bin_path> <max_cycles> <dump_base_hex> <dump_size_hex> <dump_file_path>
#include <iostream>
#include <string>
#include <cstdlib>
#include "../../include/SIMT.h"
#include "../../include/memory.h"
#include "../basic/loader.h"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <bin_path> <max_cycles> <dump_base_hex> <dump_size_hex> <dump_file_path>\n";
        return 1;
    }
    std::string bin_path = argv[1];
    long long max_cycles = std::atoll(argv[2]);
    uint32_t dump_base = std::strtoul(argv[3], nullptr, 16);
    uint32_t dump_size = std::strtoul(argv[4], nullptr, 16);
    std::string dump_path = argv[5];

    // Same 4MB RAM as tests/python/link.ld's RAM region (the kernel is
    // linked against that same linker script).
    Memory memory(4 * 1024 * 1024);
    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load " << bin_path << "\n";
        return 1;
    }

    SIMTCore core(&memory);
    // All T lanes start active; a kernel that needs fewer (e.g. a vector
    // shorter than T) narrows itself via simt_tmc() -- see
    // tests/simt/c_tests/*.c.
    core.reset_warp(0, 0x0);

    long long cycles = 0;
    while (!core.warp_halted(0) && cycles < max_cycles) {
        core.issue(0);
        cycles++;
    }

    std::cout << "Ran " << cycles << " cycles, halted=" << core.warp_halted(0) << "\n";
    memory.dump_range(dump_base, dump_base + dump_size, dump_path);
    return 0;
}
