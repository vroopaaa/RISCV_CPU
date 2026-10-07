#include <iostream>
#include <string>
#include <cstdlib>
#include "../../include/CPU.h"
#include "../../include/GridLauncher.h"
#include "../../include/memory.h"
#include "loader.h"

// Defaults, used when the optional dump-region args below aren't given.
// Points at npu_matmul_256by256.c's RESULT_BASE_ADDR (~800KB).
static const uint32_t DEFAULT_DUMP_BASE   = 0xC8000;
static const uint32_t DEFAULT_DUMP_OFFSET = 0x0100;

int main(int argc, char* argv[]) {
    // Defaults preserve old behavior: `make run-test`/`emul` with no dump
    // args still work, just without printing a memory dump (see
    // dump_requested below) -- tests now have their own in-program way to
    // print results (e.g. the custom-0 memory-print instruction), so an
    // unrequested raw hex dump of an arbitrary region is no longer forced
    // on every run.
    std::string bin_path = "tests/basic/test.bin";
    long long num_cycles = 500;
    uint32_t dump_base = DEFAULT_DUMP_BASE;
    uint32_t dump_offset = DEFAULT_DUMP_OFFSET;
    std::string dump_file_path; // empty -> dump to stdout, same as before
    bool dump_requested = argc > 3; // only dump when a region was explicitly given
    // Two new trailing args, both optional -- anything calling this with the
    // original 5 args (make run-test, run_tests.py, emul) is unaffected and
    // still runs the original single-instruction pipeline.
    std::string mode = "scalar";  // "scalar" (default) or "superscalar"
    int issue_width = 0;          // superscalar only; 0 => CPU's own default (MAX_ISSUE_WIDTH)

    if (argc > 1) bin_path = argv[1];
    if (argc > 2) num_cycles = std::atoll(argv[2]);
    if (argc > 3) dump_base = std::strtoul(argv[3], nullptr, 16);
    if (argc > 4) dump_offset = std::strtoul(argv[4], nullptr, 16);
    if (argc > 5) dump_file_path = argv[5];
    if (argc > 6) mode = argv[6];
    if (argc > 7) issue_width = std::atoi(argv[7]);
    bool superscalar = (mode == "superscalar");

    std::cout << "Starting RISC-V CPU Emulator - Compiled C Program Test..." << std::endl;
    std::cout << "Loading: " << bin_path << " | Cycles: " << num_cycles
               << " | Mode: " << mode << std::endl;

    // 4MB: matches tests/python/link.ld's LENGTH = 4M (bumped from 1M so
    // larger NPU matmul tests' result region, at a fixed offset past the
    // A/B data, has room without colliding with the 1MB boundary).
    Memory memory(4 * 1024 * 1024);

    if (!load_binary(memory, bin_path, 0x0)) {
        std::cerr << "Failed to load test binary. Exiting.\n";
        return 1;
    }

    CPU cpu(&memory);
    // GPU for the LAUNCH opcode (0x5B) -- programs using simt_launch() work
    // here too. The CPU owns the host stack reserve and hands it to the GPU.
    GridLauncher gpu(&memory, CPU::HOST_STACK_RESERVE);
    cpu.attach_gpu(&gpu);
    if (superscalar && issue_width > 0) {
        cpu.set_issue_width(issue_width);
    }

    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    if (superscalar) {
        std::cout << "Superscalar issue width: " << cpu.issue_width() << std::endl;
        for (long long i = 0; i < num_cycles; i++) {
            cpu.fetch_n();
            cpu.decode_all();
            cpu.hazard_scan();
            cpu.execute_m();
            cpu.read_m();
            cpu.writeback_m();
            if (cpu.is_halted()) break; // program reached its `j _end` halt loop
        }
    } else {
        for (long long i = 0; i < num_cycles; i++) {
            cpu.fetch();
            cpu.decode();
            cpu.execute();
            cpu.read();
            cpu.writeback();
            if (cpu.is_halted()) break; // program reached its `j _end` halt loop
        }
    }

    std::cout << "\n--- Final State ---" << std::endl;
    cpu.print_state();

    if (dump_requested) {
        if (dump_file_path.empty()) {
            std::cout << "\n--- Memory Dump ---" << std::endl;
            memory.dump_range(dump_base, dump_base + dump_offset);
        } else {
            memory.dump_range(dump_base, dump_base + dump_offset, dump_file_path);
        }
    }

    return 0;
}
