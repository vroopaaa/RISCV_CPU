// Grid-launch harness for tests/simt/run_grid_tests.py: loads an assembled
// kernel binary, launches it over the GPU through GridLauncher::launch_grid
// (directly -- the host CPU's LAUNCH opcode is phase C), prints the launch
// result on one line and dumps a memory region for the Python driver to diff.
//
// Usage: grid_harness <bin_path> <entry_hex> <threads_per_block> <num_blocks>
//                     <args_hex> <max_issues_per_block> <dump_base_hex>
//                     <dump_size_hex> <dump_file_path> [ram_bytes_hex] [host_stack_reserve_hex]
// ram_bytes defaults to 4MB (tests/python/link.ld), host_stack_reserve to CPU::HOST_STACK_RESERVE.
// Prints: cycles=<n> timed_out=<0|1> faulted=<0|1> sm_cycles=<sm0>,<sm1>,...
//
// Host mode -- boots the scalar CPU at pc 0 with the GPU attached; the host
// program launches kernels itself through the LAUNCH opcode (0x5B):
//   grid_harness --host <bin_path> <max_cycles> <dump_base_hex> <dump_size_hex>
//                       <dump_file_path> [no-gpu]
// Prints: halted=<0|1> cpu_cycles=<n> sm_cycles=<sm0>,<sm1>,...  (sm_cycles: last launch)
#include <iostream>
#include <string>
#include <cstdlib>
#include "../../include/CPU.h"
#include "../../include/GridLauncher.h"
#include "../../include/memory.h"
#include "../basic/loader.h"

static void print_sm_cycles(const GridLauncher& gpu) {
    std::cout << " sm_cycles=";
    for (uint32_t sm = 0; sm < NUM_SMS; sm++) std::cout << (sm ? "," : "") << gpu.sm_cycles((int)sm);
    std::cout << "\n";
}

static int run_host(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " --host <bin_path> <max_cycles> <dump_base_hex>"
                  << " <dump_size_hex> <dump_file_path> [no-gpu]\n";
        return 1;
    }
    std::string bin_path  = argv[2];
    long long max_cycles  = std::atoll(argv[3]);
    uint32_t dump_base    = std::strtoul(argv[4], nullptr, 16);
    uint32_t dump_size    = std::strtoul(argv[5], nullptr, 16);
    std::string dump_path = argv[6];
    bool attach           = !(argc > 7 && std::string(argv[7]) == "no-gpu");

    Memory memory(4 * 1024 * 1024); // tests/python/link.ld's RAM
    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load " << bin_path << "\n";
        return 1;
    }
    GridLauncher gpu(&memory, CPU::HOST_STACK_RESERVE); // the CPU owns the host stack reserve
    CPU cpu(&memory);
    if (attach) cpu.attach_gpu(&gpu);

    // Same scalar loop as tests/basic/main.cpp.
    for (long long i = 0; i < max_cycles && !cpu.is_halted(); i++) {
        cpu.fetch();
        cpu.decode();
        cpu.execute();
        cpu.read();
        cpu.writeback();
    }

    std::cout << "halted=" << (cpu.is_halted() ? 1 : 0) << " cpu_cycles=" << cpu.get_cycle_count();
    print_sm_cycles(gpu);
    memory.dump_range(dump_base, dump_base + dump_size, dump_path);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--host") return run_host(argc, argv);
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " <bin_path> <entry_hex> <threads_per_block> <num_blocks>"
                  << " <args_hex> <max_issues_per_block> <dump_base_hex> <dump_size_hex> <dump_file_path>"
                  << " [ram_bytes_hex] [host_stack_reserve_hex]\n";
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
    uint32_t ram_bytes         = (argc > 10) ? std::strtoul(argv[10], nullptr, 16) : 4 * 1024 * 1024;
    uint32_t host_reserve      = (argc > 11) ? std::strtoul(argv[11], nullptr, 16) : CPU::HOST_STACK_RESERVE;

    // The host keeps the top host_reserve bytes of RAM for its stack.
    Memory memory(ram_bytes);
    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load " << bin_path << "\n";
        return 1;
    }

    GridLauncher gpu(&memory, host_reserve);
    gpu.set_max_issues_per_block(max_issues);
    uint64_t cycles = gpu.launch_grid(entry, threads_per_block, num_blocks, args);

    std::cout << "cycles=" << cycles << " timed_out=" << (gpu.timed_out() ? 1 : 0)
              << " faulted=" << (gpu.faulted() ? 1 : 0);
    print_sm_cycles(gpu);

    memory.dump_range(dump_base, dump_base + dump_size, dump_path);
    return 0;
}
