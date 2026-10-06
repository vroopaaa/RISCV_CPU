#include "../include/SIMT.h"
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>

SIMTCore::SIMTCore(Memory* mem, int sm_id) : memory(mem), sm_id(sm_id) {
    std::memset(regfile, 0, sizeof(regfile));
    // warps[] default-initializes via Warp's own member initializers
    // (active=false, halted=false, pc=0, tmask=0, empty ipdom_stack).
}

reg_t SIMTCore::read_reg(int warp, int thread, int reg) const {
    assert(warp >= 0 && warp < WARPS_RESIDENT);
    assert(thread >= 0 && thread < THREADS_PER_WARP);
    assert(reg >= 0 && reg < NUM_ARCH_REGS);
    assert(physical_addr(warp, thread, reg) < (uint32_t)PHYS_REGS);
    if (reg == 0) return 0;              // x0: hardwired zero, no storage
    return regfile[thread][warp][reg - 1];
}

void SIMTCore::write_reg(int warp, int thread, int reg, reg_t value) {
    assert(warp >= 0 && warp < WARPS_RESIDENT);
    assert(thread >= 0 && thread < THREADS_PER_WARP);
    assert(reg >= 0 && reg < NUM_ARCH_REGS);
    assert(physical_addr(warp, thread, reg) < (uint32_t)PHYS_REGS);
    if (reg == 0) return;                // writes to x0 are dropped
    regfile[thread][warp][reg - 1] = value;
}

reg_t SIMTCore::read_phys(uint32_t paddr) const {
    assert(paddr < (uint32_t)PHYS_REGS);
    uint32_t reg    = paddr % NUM_ARCH_REGS;
    uint32_t thread = (paddr / NUM_ARCH_REGS) % THREADS_PER_WARP;
    uint32_t warp   = paddr / (NUM_ARCH_REGS * THREADS_PER_WARP);
    return read_reg(warp, thread, reg);
}

void SIMTCore::dump_regs(std::ostream& os, int warp) const {
    for (int t = 0; t < THREADS_PER_WARP; t++) {
        os << "--- warp " << warp << " thread " << t << " ---\n";
        for (int r = 0; r < NUM_ARCH_REGS; r++) {
            os << "x" << std::dec << std::setfill(' ') << std::left << std::setw(2) << r
               << std::right << ": 0x" << std::hex << std::setfill('0') << std::setw(8)
               << read_reg(warp, t, r) << "  ";
            if ((r + 1) % 4 == 0) os << "\n";
        }
    }
    os << std::dec << std::setfill(' ');
}

// =============================================================================
// Execution (Phase 2/3) -- see SIMT.h for the design summary. Mirrors
// simulation/simt_model.py's validated semantics (lockstep broadcast,
// per-lane masking, shared-memory-only cross-lane communication, the
// IPDOM-stack predication model for SPLIT/JOIN) with real RV32IM decoding
// and CPU's own ALU/branch logic instead of symbolic ops.
// =============================================================================

void SIMTCore::reset_warp(int w, reg_t start_pc) {
    assert(w >= 0 && w < WARPS_RESIDENT);
    Warp& warp = warps[w];
    warp.active = true;
    warp.halted = false;
    warp.pc = start_pc;
    warp.tmask = (THREADS_PER_WARP >= 32) ? 0xFFFFFFFFu : ((1u << THREADS_PER_WARP) - 1u);
    warp.ipdom_stack.clear();
    warp.block_id = 0;       // run_block() overwrites these after reset_warp();
    warp.warp_in_block = 0;  // the plain harness path keeps the single-warp defaults
}

int SIMTCore::leader(int w) const {
    uint32_t mask = warps[w].tmask;
    for (int t = 0; t < THREADS_PER_WARP; t++)
        if (mask & (1u << t)) return t;
    return 0; // empty mask: nothing is "the leader", 0 is just a safe default
}

reg_t SIMTCore::exec_lane(int w, int thread, const CPU::InstructionFields& f, bool& out_mem_read, bool& out_mem_write) {
    out_mem_read = false;
    out_mem_write = false;
    reg_t a = read_reg(w, thread, f.rs1);
    reg_t b = read_reg(w, thread, f.rs2);

    switch (f.opcode) {
        case 0x13: case 0x33: case 0x37: case 0x17: // OP-IMM / OP (incl. M ext) / LUI / AUIPC
            return CPU::alu_exec(f, a, b, warps[w].pc);

        case 0x03: { // Load -- this lane's own address, shared Memory*
            out_mem_read = true;
            reg_t addr = a + f.imm_I;
            switch (f.funct3) {
                case 0x0: return (reg_t)(int32_t)(int8_t)memory->read_byte(addr);   // LB
                case 0x1: return (reg_t)(int32_t)(int16_t)memory->read_halfword(addr); // LH
                case 0x2: return memory->read_word(addr);                            // LW
                case 0x4: return memory->read_byte(addr);                            // LBU
                case 0x5: return memory->read_halfword(addr);                        // LHU
            }
            return 0;
        }

        case 0x23: { // Store -- this lane's own address and own rs2 value
            out_mem_write = true;
            reg_t addr = a + f.imm_S;
            switch (f.funct3) {
                case 0x0: memory->write_byte(addr, b & 0xFF); break;     // SB
                case 0x1: memory->write_halfword(addr, b & 0xFFFF); break; // SH
                case 0x2: memory->write_word(addr, b); break;             // SW
            }
            return 0;
        }

        // opcode 0x0B (custom-0, NPU): Phase 5 seam, not implemented here yet --
        // falls through to the default no-op below. A kernel that issues it from
        // a SIMT lane today just gets nothing (no register write, no memory
        // access), not a crash -- see SIMT.h's Phase 5 note.
        default:
            return 0;
    }
}

void SIMTCore::exec_simt(int w, const CPU::InstructionFields& f) {
    Warp& warp = warps[w];
    switch (f.funct3) {
        case SIMT_TMC: { // rs1 = count (read from leader -- see leader()'s doc comment)
            reg_t count = read_reg(w, leader(w), f.rs1);
            if (count > (reg_t)THREADS_PER_WARP) count = THREADS_PER_WARP;
            warp.tmask = (count == 0) ? 0u : (uint32_t)((1ull << count) - 1u);
            break;
        }
        case SIMT_TID: { // rd = pack_tid(w, thread) for every active lane
            for (int t = 0; t < THREADS_PER_WARP; t++)
                if (warp.tmask & (1u << t))
                    write_reg(w, t, f.rd, pack_tid(warp.warp_in_block, (uint32_t)t));
            break;
        }
        case SIMT_IDENT:
            exec_ident(w, f);
            break;
        case SIMT_SPLIT: { // rs1 = predicate reg, rs2 = reg holding reconv_pc
            uint32_t old_mask = warp.tmask;
            uint32_t taken_mask = 0;
            for (int t = 0; t < THREADS_PER_WARP; t++)
                if ((old_mask & (1u << t)) && read_reg(w, t, f.rs1) != 0)
                    taken_mask |= (1u << t);
            reg_t reconv_pc = read_reg(w, leader(w), f.rs2);
            warp.ipdom_stack.push_back({old_mask, reconv_pc});
            warp.tmask = taken_mask;
            break;
        }
        case SIMT_JOIN: {
            assert(!warp.ipdom_stack.empty() && "SIMT JOIN with an empty ipdom_stack (unbalanced SPLIT/JOIN)");
            IpdomEntry entry = warp.ipdom_stack.back();
            warp.ipdom_stack.pop_back();
            warp.tmask = entry.mask;
            break;
        }
        case SIMT_PRED: { // rs1 = predicate reg; narrows tmask with no stack push (caller restores manually)
            uint32_t new_mask = 0;
            for (int t = 0; t < THREADS_PER_WARP; t++)
                if ((warp.tmask & (1u << t)) && read_reg(w, t, f.rs1) != 0)
                    new_mask |= (1u << t);
            warp.tmask = new_mask;
            break;
        }
        case SIMT_WSPAWN:
        case SIMT_BAR:
        default:
            // Not implemented yet -- both are inherently multi-warp (WSPAWN
            // activates OTHER warps; BAR needs several warps to rendezvous
            // against), so they need the Phase 4 round-robin scheduler to mean
            // anything. A single-warp issue() can't do either meaningfully yet.
            break;
    }
}

void SIMTCore::exec_ident(int w, const CPU::InstructionFields& f) {
    Warp& warp = warps[w];
    for (int t = 0; t < THREADS_PER_WARP; t++) {
        if (!(warp.tmask & (1u << t))) continue;
        switch ((uint8_t)f.funct7) {
            case IDENT_BLOCK_IDX: write_reg(w, t, f.rd, warp.block_id);   break;
            case IDENT_BLOCK_DIM: write_reg(w, t, f.rd, block_dim);       break;
            case IDENT_GRID_DIM:  write_reg(w, t, f.rd, grid_dim);        break;
            case IDENT_HW_TID:    write_reg(w, t, f.rd, hw_tid(w, t));    break;
            default: break; // unknown selector: no-op, same posture as the other unimplemented SIMT ops
        }
    }
}

// =============================================================================
// Block execution -- see docs/CUDA/grid_launch_plan.md section 5.1.
// =============================================================================

uint32_t SIMTCore::hw_tid(int w, int lane) const {
    return ((uint32_t)sm_id * WARPS_RESIDENT + (uint32_t)w) * THREADS_PER_WARP + (uint32_t)lane;
}

reg_t SIMTCore::lane_stack_top(const BlockLaunch& b, int w, int lane) const {
    return b.stack_top - (hw_tid(w, lane) + 1) * STACK_BYTES_PER_THREAD;
}

void SIMTCore::init_block_warp(int w, const BlockLaunch& b, int nwarps) {
    reset_warp(w, b.entry);
    // Last warp of a block whose size isn't a multiple of T runs only its live lanes.
    if (w == nwarps - 1 && (b.block_dim % THREADS_PER_WARP) != 0)
        warps[w].tmask = (1u << (b.block_dim % THREADS_PER_WARP)) - 1u;

    // Deterministic start: no stale registers from whatever ran in this slot before.
    for (int t = 0; t < THREADS_PER_WARP; t++)
        for (int r = 1; r < NUM_ARCH_REGS; r++)
            write_reg(w, t, r, 0);

    // Runtime-provided calling state for each live lane: a kernel is a plain
    // C function void kernel(void* args) -- sp = private stack, ra = the
    // sentinel that ends the warp, a0 = args.
    for (int t = 0; t < THREADS_PER_WARP; t++) {
        if (!(warps[w].tmask & (1u << t))) continue;
        write_reg(w, t, 2,  lane_stack_top(b, w, t)); // sp
        write_reg(w, t, 1,  BLOCK_RETURN_PC);         // ra
        write_reg(w, t, 10, b.args);                  // a0
    }
    warps[w].block_id = b.block_id;
    warps[w].warp_in_block = (uint32_t)w;
}

uint64_t SIMTCore::run_block(const BlockLaunch& b, bool* timed_out, uint64_t max_issues) {
    if (timed_out) *timed_out = false;
    fault_raised = false;
    if (b.block_dim == 0 || b.block_dim > MAX_THREADS_PER_BLOCK) {
        std::cerr << "[SIMTCore Error] block_dim " << b.block_dim << " out of range 1.."
                  << MAX_THREADS_PER_BLOCK << "\n";
        return 0;
    }

    block_dim = b.block_dim;
    grid_dim  = b.grid_dim;
    int nwarps = (int)((b.block_dim + THREADS_PER_WARP - 1) / THREADS_PER_WARP);

    for (int w = 0; w < nwarps; w++)
        init_block_warp(w, b, nwarps);

    // Run-to-completion, warp by warp: no latency in the model, so any
    // interleaving gives the same result for barrier-free kernels.
    uint64_t issued = 0;
    bool overran = false;
    for (int w = 0; w < nwarps && !overran && !fault_raised; w++) {
        while (!warps[w].halted) {
            if (issued >= max_issues) { overran = true; break; }
            issue(w);
            issued++; // a faulting instruction still counts as issued
        }
    }
    if (overran) {
        std::cerr << "[SIMTCore Error] SM " << sm_id << " block " << b.block_id
                  << " exceeded " << max_issues << " issued instructions -- abandoned\n";
        if (timed_out) *timed_out = true;
    }

    for (int w = 0; w < nwarps; w++) warps[w].active = false; // leave the slots clean
    return issued;
}

// Everything an SM lane can run is listed here; anything else faults instead
// of being silently skipped (a skipped instruction just gives wrong results).
const char* SIMTCore::unsupported_name(const CPU::InstructionFields& f) {
    switch (f.opcode) {
        case 0x13: case 0x33: case 0x37: case 0x17: // OP-IMM / OP (incl. M ext) / LUI / AUIPC
        case 0x6F:                                   // JAL
            return nullptr;
        case 0x0F:                                   // FENCE: no reordering in this model, a no-op is correct
            return nullptr;
        case 0x67:                                   // JALR
            return (f.funct3 == 0) ? nullptr : "JALR";
        case 0x03:                                   // LB/LH/LW/LBU/LHU
            return (f.funct3 <= 2 || f.funct3 == 4 || f.funct3 == 5) ? nullptr : "LOAD";
        case 0x23:                                   // SB/SH/SW
            return (f.funct3 <= 2) ? nullptr : "STORE";
        case 0x63:                                   // BEQ/BNE/BLT/BGE/BLTU/BGEU
            return (f.funct3 == 2 || f.funct3 == 3) ? "BRANCH" : nullptr;
        case OPCODE_SIMT:
            switch (f.funct3) {
                case SIMT_WSPAWN: return "SIMT_WSPAWN"; // needs the Phase 4 scheduler
                case SIMT_BAR:    return "SIMT_BAR";    // needs the Phase 4 scheduler
                case SIMT_IDENT:  return ((uint8_t)f.funct7 <= IDENT_HW_TID) ? nullptr : "SIMT_IDENT";
                default:          return nullptr;
            }
        case CPU::OPCODE_LAUNCH: return "LAUNCH";   // no launching from inside a kernel
        case 0x0B:               return "NPU";      // GPU -> NPU handoff not built (Phase 5)
        case 0x73:               return "SYSTEM";   // ECALL/EBREAK/CSR: no trap or CSR support
        default:                 return "unknown";
    }
}

void SIMTCore::issue(int w) {
    assert(w >= 0 && w < WARPS_RESIDENT);
    Warp& warp = warps[w];
    if (!warp.active || warp.halted) return;

    reg_t word = memory->read_word(warp.pc);
    CPU::InstructionFields f = CPU::decode_one(word);
    reg_t pc = warp.pc;
    reg_t next_pc = pc + 4; // default fall-through, same as CPU::fetch()

    const char* bad = unsupported_name(f);
    if (bad != nullptr) {
        std::cerr << "[SIMTCore Error] SM " << sm_id << " block " << warp.block_id << " warp " << w
                  << " pc 0x" << std::hex << pc << ": unsupported instruction 0x" << std::setw(8)
                  << std::setfill('0') << word << std::setfill(' ') << std::dec << " (" << bad << ")\n";
        fault_raised = true;
        warp.halted = true; // pc stays on the faulting instruction
        return;
    }

    if (f.opcode == OPCODE_SIMT) {
        exec_simt(w, f); // warp-level; none of the SIMT ops touch pc

    } else if (f.opcode == 0x63) { // BRANCH -- SIMT-uniform only, see CPU::branch_taken
        bool have_decision = false, taken = false;
        for (int t = 0; t < THREADS_PER_WARP; t++) {
            if (!(warp.tmask & (1u << t))) continue;
            bool lane_taken = CPU::branch_taken(f.funct3, read_reg(w, t, f.rs1), read_reg(w, t, f.rs2));
            if (!have_decision) { taken = lane_taken; have_decision = true; }
            else assert(lane_taken == taken &&
                        "SIMT BRANCH divergence: active lanes disagree -- use SPLIT/JOIN "
                        "for a condition that can differ per lane");
        }
        if (taken) {
            next_pc = pc + f.imm_B;
            if (next_pc == pc) warp.halted = true; // same halt idiom as CPU: a branch targeting itself
        }

    } else if (f.opcode == 0x6F) { // JAL -- fully uniform (target has no register inputs at all)
        reg_t link = pc + 4;
        for (int t = 0; t < THREADS_PER_WARP; t++)
            if (warp.tmask & (1u << t)) write_reg(w, t, f.rd, link);
        next_pc = pc + f.imm_J;
        if (next_pc == pc) warp.halted = true;

    } else if (f.opcode == 0x67) { // JALR -- target computed from the leader lane's rs1 only
                                    // (unlike BRANCH, no cross-lane agreement check -- real SIMT
                                    // kernels essentially never compute a per-lane-divergent JALR
                                    // target; flagged here rather than silently assumed correct)
        reg_t link = pc + 4;
        for (int t = 0; t < THREADS_PER_WARP; t++)
            if (warp.tmask & (1u << t)) write_reg(w, t, f.rd, link);
        next_pc = (read_reg(w, leader(w), f.rs1) + f.imm_I) & ~1u;
        if (next_pc == pc) warp.halted = true;

    } else {
        // Lane-broadcast path: ALU (incl. M ext)/LUI/AUIPC/loads write rd;
        // stores and everything else (incl. the unimplemented NPU seam) don't.
        bool writes_rd = (f.opcode == 0x13 || f.opcode == 0x33 || f.opcode == 0x37 ||
                           f.opcode == 0x17 || f.opcode == 0x03);
        for (int t = 0; t < THREADS_PER_WARP; t++) {
            if (!(warp.tmask & (1u << t))) continue;
            bool mem_read, mem_write;
            reg_t result = exec_lane(w, t, f, mem_read, mem_write);
            (void)mem_read; (void)mem_write; // control-signal shape kept for symmetry with CPU; unused here
            if (writes_rd) write_reg(w, t, f.rd, result);
        }
    }

    warp.pc = next_pc;
    if (warp.pc == BLOCK_RETURN_PC) warp.halted = true; // kernel function returned (ra sentinel)
}
