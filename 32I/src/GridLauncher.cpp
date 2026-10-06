#include "../include/GridLauncher.h"
#include <iostream>

//Constructor
GridLauncher::GridLauncher(Memory* mem, uint32_t host_stack_reserve)
    : memory(mem), cycles_per_sm(NUM_SMS, 0) {
    // Reserve space for the host stack; every SM's device stacks go right below it.
    const uint64_t device_stacks = (uint64_t)NUM_SMS * SIMTCore::MAX_THREADS_PER_BLOCK * SIMTCore::STACK_BYTES_PER_THREAD;
    if ((uint64_t)mem->size() < (uint64_t)host_stack_reserve + device_stacks) {
        std::cerr << "[GridLauncher Error] RAM " << mem->size() << " bytes can't hold the host stack reserve ("
                  << host_stack_reserve << ") plus the device stacks (" << device_stacks
                  << ") -- every launch will be refused\n";
    } else {
        device_stack_top = (reg_t)(mem->size() - host_stack_reserve);
        stacks_fit = true;
    }
    //Initialise the grid: one SIMTCore per SM, each with its own sm_id so
    //hw_tid / per-lane stacks never collide across SMs.
    sm_cores.reserve(NUM_SMS);
    for (uint32_t i = 0; i < NUM_SMS; i++) {
        sm_cores.emplace_back(memory, (int)i);
    }
}

//Grid Launcher only needs the kernel entry, threads per block, the number of blocks and the kernel args

// Launches the grid of blocks on the GPU
uint64_t GridLauncher::launch_grid(reg_t entry, uint32_t threads_per_block, uint32_t nBlocks, reg_t args) {
    for (uint32_t sm = 0; sm < NUM_SMS; sm++) cycles_per_sm[sm] = 0; // fresh counts for this launch
    this->threads_per_block = 0;
    this->nBlocks = 0;
    grid_timed_out = false;

    if (!stacks_fit) {
        std::cerr << "[GridLauncher Error] launch refused: no room for the device stacks\n";
        return 0;
    }

    if (threads_per_block == 0 || threads_per_block > SIMTCore::MAX_THREADS_PER_BLOCK) {
        std::cerr << "[GridLauncher Error] threads_per_block " << threads_per_block
                  << " out of range 1.." << SIMTCore::MAX_THREADS_PER_BLOCK << "\n";
        return 0;
    }
    if (nBlocks == 0 || nBlocks > MAX_BLOCKS) {
        std::cerr << "[GridLauncher Error] nBlocks " << nBlocks
                  << " out of range 1.." << MAX_BLOCKS << "\n";
        return 0;
    }
    this->threads_per_block = threads_per_block;
    this->nBlocks = nBlocks;

    // Sequential in the emulator; "parallel" only in the cycle accounting.
    for (uint32_t i = 0; i < nBlocks; i++) {
        uint32_t sm = 0; // Round-robin assignment of blocks to SMs
        SIMTCore::BlockLaunch block_info{
            entry,
            i,                 // block_id
            threads_per_block, // block_dim
            nBlocks,           // grid_dim
            args,
            device_stack_top,
        };
        bool block_timed_out = false;
        cycles_per_sm[sm] += sm_cores[sm].run_block(block_info, &block_timed_out, max_issues_per_block);
        if (block_timed_out) {
            // A runaway kernel would burn the cap again on every remaining block.
            std::cerr << "[GridLauncher Error] block " << i << " timed out -- abandoning the remaining "
                      << (nBlocks - 1 - i) << " blocks\n";
            grid_timed_out = true;
            break;
        }
    }

    if (verbose) status_grid();
    return max_sm_cycles();
}

// Device cycles of the last launch: the busiest SM's total (SMs run in parallel).
uint64_t GridLauncher::max_sm_cycles() const {
    uint64_t max_cycles = 0;
    for (uint32_t sm = 0; sm < NUM_SMS; sm++)
        if (cycles_per_sm[sm] > max_cycles) max_cycles = cycles_per_sm[sm];
    return max_cycles;
}

// Prints the status of the entire grid
void GridLauncher::status_grid() const {
    std::cout << "[GridLauncher] " << nBlocks << " blocks x " << threads_per_block
              << " threads on " << NUM_SMS << " SMs, device cycles "
              << max_sm_cycles() << "\n";
    for (uint32_t sm = 0; sm < NUM_SMS; sm++) status_sm((int)sm);
}

// Prints the status of a specific SM
void GridLauncher::status_sm(int sm_id) const {
    std::cout << "  SM " << sm_id << ": cycles " << cycles_per_sm[sm_id] << ", blocks";
    for (uint32_t b = (uint32_t)sm_id; b < nBlocks; b += NUM_SMS) std::cout << " " << b;
    std::cout << "\n";
}
