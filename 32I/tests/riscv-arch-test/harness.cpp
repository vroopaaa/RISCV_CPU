// Dedicated harness for riscv-arch-test-style signature tests. Loads a raw
// binary, runs the CPU until it hits its halt idiom (or a cycle cap, as a
// safety net), then dumps the memory word range [sig_start, sig_end) to
// stdout as one lowercase hex word per line -- run_arch_test.py parses this
// and diffs it against an independently computed Python reference.
#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include "../../include/CPU.h"
#include "../../include/memory.h"
#include "../basic/loader.h"

// Must match link_arch_test.ld's MEMORY ORIGIN. Non-zero because the same
// binary also runs under qemu-riscv32 (as a real Linux process, via
// QEMU_REFERENCE builds) as the reference for run_arch_test.py, and
// qemu-riscv32 refuses to mmap a LOAD segment at address 0x0.
static const uint32_t LINK_ORIGIN = 0x00010000;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <bin_path> <sig_start_hex> <sig_end_hex> [max_cycles]\n";
        return 1;
    }

    std::string bin_path = argv[1];
    uint32_t sig_start = std::strtoul(argv[2], nullptr, 16);
    uint32_t sig_end = std::strtoul(argv[3], nullptr, 16);
    int max_cycles = (argc > 4) ? std::atoi(argv[4]) : 200000;

    Memory memory(1024 * 1024);
    if (!load_binary(memory, bin_path, LINK_ORIGIN)) {
        std::cerr << "Failed to load test binary. Exiting.\n";
        return 1;
    }

    CPU cpu(&memory);
    cpu.set_pc(LINK_ORIGIN);
    for (int i = 0; i < max_cycles; i++) {
        cpu.fetch();
        cpu.decode();
        cpu.execute();
        cpu.read();
        cpu.writeback();
        if (cpu.is_halted()) break;
    }

    for (uint32_t addr = sig_start; addr < sig_end; addr += 4) {
        std::cout << std::hex << std::setfill('0') << std::setw(8)
                   << memory.read_word(addr) << "\n";
    }

    return 0;
}
