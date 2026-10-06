#ifndef SIMT_H
#define SIMT_H

#include <cstdint>
#include <ostream>
#include <vector>
#include "CPU.h"      // reg_t, and SIMTCore is a friend of CPU for decode_one/alu_exec/branch_taken
#include "memory.h"

// =============================================================================
// SIMT execution layer.
//   Phase 1: banked register file (done).
//   Phase 2/3: this increment -- single-warp lane-broadcast execution engine
//     (ALU/load/store/branch) plus the custom-1 (0x2B) SIMT control opcodes
//     TMC/TID/SPLIT/JOIN/PRED. Mirrors simulation/simt_model.py's validated
//     semantics 1:1 (see that file's docstrings for the design rationale --
//     this is the same model, just with real RV32IM encoding/decoding
//     instead of symbolic ops, and reusing CPU's decode/ALU/branch logic via
//     friendship instead of reimplementing it a third time).
//   Not yet implemented here: WSPAWN/BAR (inherently multi-warp -- need the
//     Phase 4 round-robin scheduler to mean anything) and the NPU handoff
//     seam (Phase 5).
// Sits alongside -- not inside -- the scalar/superscalar CPU class.
// =============================================================================

class SIMTCore {
public:
    // ---- Hardcoded configuration (not build-configurable yet) ----
    // THREADS_PER_WARP bumped 4 -> 32 (matches real NVIDIA warp size) for the
    // 1-SM/32-thread ReLU experiment -- see docs/CUDA/progress.md. The
    // engine itself never hardcoded 4 anywhere (tmask is uint32_t, and
    // reset_warp/exec_simt already guarded the T==32 shift-by-32 edge case
    // from day one), so this was a one-line change with no other code
    // affected -- only test files that had their own hardcoded loop bounds
    // needed updating (see tests/simt/).
    static constexpr int THREADS_PER_WARP = 32;  // T
    static constexpr int WARPS_RESIDENT   = 4;   // W
    static constexpr int NUM_ARCH_REGS    = 32;
    static constexpr int IPDOM_DEPTH      = 8;    // documented bound on nested SPLITs;
                                                    // not enforced (ipdom_stack is a vector) --
                                                    // matches simulation/simt_model.py's IPDOM_DEPTH.
    static_assert(THREADS_PER_WARP <= 32, "tmask is a uint32_t");

    // ---- Block execution (grid launch -- see docs/CUDA/grid_launch_plan.md) ----
    // A kernel function's `ret` jumps here (the launcher seeds every lane's ra
    // with it); issue() treats reaching it as "this warp is done". Outside RAM.
    static constexpr uint32_t BLOCK_RETURN_PC        = 0xFFFFFFF0u;
    // Private stack slice per lane, carved below BlockLaunch::stack_top.
    static constexpr uint32_t STACK_BYTES_PER_THREAD = 0x400;
    // A block must fit one SM's warp slots.
    static constexpr uint32_t MAX_THREADS_PER_BLOCK  = THREADS_PER_WARP * WARPS_RESIDENT;

    // Total architectural registers across all resident threads: T * W * 32
    // (512 at T=4, W=4).
    static constexpr int PHYS_REGS = THREADS_PER_WARP * WARPS_RESIDENT * NUM_ARCH_REGS;

    // Custom-1 opcode (same R-type shape as the NPU's custom-0: `.insn r
    // 0x2B, funct3, 0, rd, rs1, rs2`) and its funct3 sub-ops.
    static constexpr uint8_t OPCODE_SIMT = 0x2B;
    enum SimtOp : uint8_t {
        SIMT_TMC    = 0, // rs1 = count -> tmask = (1 << count) - 1 for the current warp
        SIMT_WSPAWN = 1, // not implemented yet -- needs the Phase 4 scheduler
        SIMT_TID    = 2, // rd = pack_tid(warp, thread) for the calling thread
        SIMT_BAR    = 3, // not implemented yet -- needs the Phase 4 scheduler
        SIMT_SPLIT  = 4, // rs1 = predicate reg, rs2 = reg holding reconv_pc (read from the
                         // leader lane -- see leader()); push (old_mask, reconv_pc), narrow
                         // tmask to lanes where predicate != 0
        SIMT_JOIN   = 5, // pop ipdom_stack, restore tmask (no operands)
        SIMT_PRED   = 6, // rs1 = predicate reg; AND tmask with per-lane predicate, no stack
                         // push -- "soft" narrowing the caller must widen back manually
        SIMT_IDENT  = 7, // rd = block/grid identity value, selected by funct7 (see IdentOp)
    };

    // funct7 selector for SIMT_IDENT.
    enum IdentOp : uint8_t {
        IDENT_BLOCK_IDX = 0, // blockIdx of the running block
        IDENT_BLOCK_DIM = 1, // threads per block
        IDENT_GRID_DIM  = 2, // number of blocks in the grid
        IDENT_HW_TID    = 3, // unique physical thread id: (sm*W + warp_slot)*T + lane
    };

    // TID packing: [23:16] flat thread index (warp*T+thread) | [15:8] warp_id | [7:0] thread_id.
    // `warp` here is the warp's index WITHIN ITS BLOCK (warp_in_block), so
    // [23:16] is threadIdx. For the single-warp harness path (warp 0, no
    // launch) it is the same value as the warp slot, so nothing changes there.
    // Exact bit layout is this prototype's own choice (the spec left it open) --
    // matches simulation/simt_model.py's _op_tid in spirit (lane index), extended
    // with a warp field since the C++ side actually has W resident warps.
    static constexpr uint32_t pack_tid(uint32_t warp, uint32_t thread) {
        return ((warp * THREADS_PER_WARP + thread) << 16) | (warp << 8) | thread;
    }

    // sm_id feeds IDENT_HW_TID / per-lane stack placement so several SMs sharing
    // one Memory never collide; defaults to 0 so single-SM callers are unchanged.
    explicit SIMTCore(Memory* mem, int sm_id = 0);

    // Everything one block needs, handed over by the grid launcher.
    struct BlockLaunch {
        reg_t    entry;      // kernel function address
        uint32_t block_id;
        uint32_t block_dim;  // threads per block, 1..MAX_THREADS_PER_BLOCK
        uint32_t grid_dim;   // number of blocks in the grid
        reg_t    args;       // goes into a0 of every lane
        reg_t    stack_top;  // top of the device-stack area; each lane's slice is carved below it
    };

    // ---- Register file (Phase 1) ----
    // Storage is banked BY LANE:  regfile[thread_lane][warp_id][reg - 1]
    // so all T lanes can read their operands for the same warp in one cycle
    // (T parallel bank reads, one per bank -- lane t only ever touches bank t).
    // x0 is hardwired to 0 for every thread and has no storage: reads return
    // 0, writes are dropped. That is why each (lane, warp) row stores 31
    // entries (x1..x31) even though the flat address space below reserves 32
    // slots per thread.f
    reg_t read_reg(int warp, int thread, int reg) const;
    void  write_reg(int warp, int thread, int reg, reg_t value);

    // Flat "physical" address of (warp, thread, reg):
    //   warp * T * 32 + thread * 32 + reg
    // This is a logical numbering only -- it is NOT how the storage is laid
    // out (see above). Useful for debug dumps and assertions.
    static constexpr uint32_t physical_addr(uint32_t warp, uint32_t thread, uint32_t reg) {
        return warp * THREADS_PER_WARP * NUM_ARCH_REGS + thread * NUM_ARCH_REGS + reg;
    }
    // Inverse mapping: read by flat physical address (debug/assert use only).
    reg_t read_phys(uint32_t paddr) const;

    // Prints every thread's registers for one warp (4 regs per row).
    void dump_regs(std::ostream& os, int warp) const;

    // ---- Execution (Phase 2/3) ----
    // Resets warp `w` to active, all T lanes on, pc = start_pc, empty
    // ipdom_stack. Every other warp is left exactly as it was (does NOT
    // touch its registers) -- for now, with no WSPAWN, a harness calls this
    // directly per warp it wants to drive instead of a real kernel launch.
    void reset_warp(int w, reg_t start_pc);

    // Runs one block to completion on this SM: sets up ceil(block_dim/32) warps
    // (last one partially masked), then runs them one after another -- no
    // scheduler, there is no latency to hide yet. Returns instructions issued.
    // A block that exceeds max_issues is abandoned with an error message (and
    // *timed_out set) so a runaway kernel can't hang the host.
    uint64_t run_block(const BlockLaunch& b, bool* timed_out = nullptr,
                       uint64_t max_issues = 1000000);

    // Fetches+executes ONE instruction for warp w: ALU/branch ops reuse
    // CPU::alu_exec/branch_taken via friendship, broadcasting to every
    // lane with its tmask bit set; loads/stores go through the shared
    // Memory* at each active lane's own address; the custom-1 SIMT ops
    // (see SimtOp) are warp-level, not lane-broadcast. Advances warp.pc by
    // one instruction unless a BRANCH/JAL/JALR overrides it (same halt
    // idiom as CPU: a jump/branch targeting its own address sets halted).
    void issue(int w);

    bool     warp_active(int w) const  { return warps[w].active; }
    bool     warp_halted(int w) const  { return warps[w].halted; }
    uint32_t warp_pc(int w) const      { return warps[w].pc; }
    uint32_t warp_tmask(int w) const   { return warps[w].tmask; }
    size_t   warp_ipdom_depth(int w) const { return warps[w].ipdom_stack.size(); }

    // True once a warp has hit an instruction an SM can't run (see
    // unsupported_name()). issue() prints the error and halts that warp;
    // run_block() then stops the whole block. Cleared at the start of each run_block().
    bool     faulted() const           { return fault_raised; }

private:
    struct IpdomEntry { uint32_t mask; reg_t reconv_pc; };
    struct Warp {
        bool     active = false;
        bool     halted = false;
        reg_t    pc = 0;
        uint32_t tmask = 0;
        std::vector<IpdomEntry> ipdom_stack;
        uint32_t block_id = 0;       // which block this warp belongs to (run_block)
        uint32_t warp_in_block = 0;  // this warp's index within its block
    };

    Memory* memory;
    int      sm_id;
    // Dimensions of the block currently running; the defaults make the
    // identity ops sane for the old single-warp harness path (no launch).
    uint32_t block_dim = THREADS_PER_WARP;
    uint32_t grid_dim  = 1;
    reg_t regfile[THREADS_PER_WARP][WARPS_RESIDENT][NUM_ARCH_REGS - 1];
    Warp  warps[WARPS_RESIDENT];
    bool  fault_raised = false;

    // nullptr if an SM lane can run f; otherwise a short name for the error
    // message (e.g. "LAUNCH", "NPU", "SIMT_BAR", "LOAD" for a bad width).
    static const char* unsupported_name(const CPU::InstructionFields& f);

    // Lowest-numbered active lane in warp w (0 if its mask is empty) --
    // used wherever a warp-level op needs to read "the" value of a register
    // that's assumed uniform across active lanes (e.g. SPLIT's reconv_pc).
    int leader(int w) const;

    // Lane-level sub-dispatch, called once per active lane by issue() for
    // everything that ISN'T a custom-1 SIMT op: ALU (incl. M ext)/LUI/
    // AUIPC via CPU::alu_exec, loads/stores via memory. Returns the ALU
    // result (loads overwrite it with the loaded value); writeback is the
    // caller's job (issue() decides whether rd gets written, since that's
    // uniform across lanes -- same control-signal shape as CPU::decode()).
    reg_t exec_lane(int w, int thread, const CPU::InstructionFields& f, bool& out_mem_read, bool& out_mem_write);

    // Custom-1 (0x2B) warp-level dispatch -- called ONCE per cycle, not per
    // lane (see SimtOp). Returns true if it set warp.pc itself (none of
    // them do today, but keeps the same contract issue() already uses for
    // BRANCH/JAL/JALR).
    void exec_simt(int w, const CPU::InstructionFields& f);

    // SIMT_IDENT handler: rd = the value selected by f.funct7 (IdentOp) for
    // every active lane; HW_TID is the only one that differs per lane.
    void exec_ident(int w, const CPU::InstructionFields& f);

    // Unique physical thread id across all SMs: (sm_id*W + warp_slot)*T + lane.
    uint32_t hw_tid(int w, int lane) const;
    // This lane's initial sp: b.stack_top - (hw_tid+1)*STACK_BYTES_PER_THREAD.
    reg_t lane_stack_top(const BlockLaunch& b, int w, int lane) const;
    // Prepares warp slot w for block b: reset, mask the partial last warp,
    // zero its registers, seed sp/ra/a0 per live lane, record block context.
    void init_block_warp(int w, const BlockLaunch& b, int nwarps);
};

#endif // SIMT_H
