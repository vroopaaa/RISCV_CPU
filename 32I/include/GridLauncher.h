#ifndef GRIDLAUNCHER_H
#define GRIDLAUNCHER_H

#include <cstdint>
#include <vector>
#include "memory.h"
#include "SIMT.h"
#include "CPU.h"

// Grid launcher -- see docs/CUDA/grid_launch_plan.md, section 5.2.
static constexpr uint32_t NUM_SMS    = 4;     // Number of Streaming Multiprocessors (SMs) in the GPU
static constexpr uint32_t MAX_BLOCKS = 65535; // Maximum number of blocks that can be launched (16-bit LAUNCH field)
// Threads per block are capped by SIMTCore::MAX_THREADS_PER_BLOCK (a block must fit one SM's warp slots).

class GridLauncher {
    public:
        // host_stack_reserve is owned by the host CPU (CPU::HOST_STACK_RESERVE):
        // the top host_stack_reserve bytes of RAM are left to the host's stack,
        // and the device stacks are carved top-down from just below them. RAM too
        // small for the reserve plus every SM's stacks is reported here and every
        // launch_grid() is then refused.
        GridLauncher(Memory* mem, uint32_t host_stack_reserve);

        // Runs every block of the grid to completion. Block b goes to SM
        // b % NUM_SMS (round-robin); blocks on one SM run back to back. Returns
        // the device cycles: the busiest SM's total, since SMs run in parallel.
        // An invalid launch prints an error and returns 0 without running anything.
        uint64_t launch_grid(reg_t entry, uint32_t threads_per_block, uint32_t nBlocks, reg_t args);

        void set_verbose(bool v) { verbose = v; }   // print status_grid() after every launch
        // Per-block issue cap handed to SIMTCore::run_block. The first block to
        // hit it abandons the rest of the grid (timed_out() then reports true).
        void set_max_issues_per_block(uint64_t n) { max_issues_per_block = n; }
        void status_grid() const;                   // block->SM map + per-SM cycles of the last launch
        void status_sm(int sm_id) const;            // one SM's blocks + cycles of the last launch

        reg_t    stack_top() const { return device_stack_top; }
        uint64_t sm_cycles(int sm_id) const { return cycles_per_sm[sm_id]; } // last launch
        bool     timed_out() const { return grid_timed_out; }                 // last launch

    private:
        uint64_t max_sm_cycles() const;      // largest entry of cycles_per_sm
        Memory* memory;
        reg_t device_stack_top = 0;
        bool stacks_fit = false;             // RAM holds the host reserve + all device stacks
        uint32_t threads_per_block = 0;
        uint32_t nBlocks = 0;
        bool verbose = false;
        bool grid_timed_out = false;
        uint64_t max_issues_per_block = 1000000; // run_block's default cap
        std::vector<SIMTCore> sm_cores;      // the SMs, all sharing one Memory*
        std::vector<uint64_t> cycles_per_sm; // per-SM issued instructions, last launch
};

#endif // GRIDLAUNCHER_H
