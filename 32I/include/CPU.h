#ifndef CPU_H
#define CPU_H

#include <cstdint>
#include "memory.h"

// Define the architecture parameters
#define XLEN 32
#define NUM_REGS 32

typedef uint32_t reg_t;

class GridLauncher; // include/GridLauncher.h -- forward-declared: SIMT.h includes CPU.h

class CPU {
    // SIMTCore (32I/include/SIMT.h) reuses CPU's RV32IM decode/ALU/branch
    // logic for its own lanes instead of duplicating it a third time (see
    // decode_one/alu_exec/branch_taken below) -- that's the only reason
    // this friendship exists. InstructionFields and these helpers stay
    // private/not part of CPU's public interface otherwise.
    friend class SIMTCore;
private:
    reg_t registers[NUM_REGS]; // The 32 general-purpose registers
    reg_t pc;                  // The Program Counter
    reg_t next_pc;             // The next Program Counter (for branch handling)
    reg_t instruction;          // The current instruction being executed
    reg_t aluResult;            // Result from the ALU operation
    reg_t memResult;            // Result from memory read operation
    uint64_t cycle_count;       // Counts completed cycles (one per fetch..writeback pass, plus
                                 // a LAUNCH's device cycles); separate from the 32 GPRs, not
                                 // addressable by any instruction. 64-bit: a grid launch can
                                 // add more cycles than fit in 32 bits.
    bool halted;                // Set by execute() when a jump/branch targets its own address
                                 // (this codebase's halt idiom, e.g. start.s's `_end: j _end`)
    Memory* memory;            // Pointer to the memory object
    GridLauncher* gpu;         // GPU run by the LAUNCH opcode (0x5B); null = no GPU attached
    // Helper method to enforce hardware rules
    void enforce_zero_register();
    bool isBranchTaken();
    reg_t branchPC;
    bool mem_read_enable;
    bool mem_write_enable;
    bool reg_write_enable;
    struct InstructionFields {
        uint8_t opcode;
        uint8_t rd;
        uint8_t rs1;
        uint8_t rs2;
        uint8_t funct3;
        int8_t funct7;
        int32_t imm_U;
        int32_t imm_I;
        int32_t imm_S;
        int32_t imm_B;
        int32_t imm_J;
        int32_t imm_Z;
    };
    InstructionFields decodedInstruction;

    static constexpr int MAX_ISSUE_WIDTH = 10;
    int issueWidth;                                    // n: configured fetch/issue window size, <= MAX_ISSUE_WIDTH
    int issueCount;                                     // m: instructions actually issued this cycle, set by hazard_scan()
    int fetchCount;                                     // instructions fetched this cycle (<= issueWidth)
    reg_t windowBasePC;                                  // pc at the start of this cycle's window (== pc when fetch_n() runs)
    reg_t windowNextPC;                                   // next_pc decided by execute_m(), committed by writeback_m()
    reg_t fetchNextPC;                                   // next PC after all fetched slots if no mispredict
    reg_t windowWords[MAX_ISSUE_WIDTH];                    // raw 32-bit words, filled by fetch_n()
    reg_t windowSlotPC[MAX_ISSUE_WIDTH];                   // per-slot PC, filled by fetch_n()
    bool windowIsBranch[MAX_ISSUE_WIDTH];                  // true if slot is conditional branch (0x63)
    bool windowPredictedTaken[MAX_ISSUE_WIDTH];            // predicted taken/not-taken
    reg_t windowPredictedTarget[MAX_ISSUE_WIDTH];          // predicted target PC
    bool windowIsSpeculative[MAX_ISSUE_WIDTH];             // true if fetched downstream of a branch
    bool windowCancelled[MAX_ISSUE_WIDTH];                 // true if squashed on misprediction
    InstructionFields windowDecoded[MAX_ISSUE_WIDTH];       // filled by decode_all()
    bool windowMemRead[MAX_ISSUE_WIDTH];                     // per-slot control signals, filled by decode_all()
    bool windowMemWrite[MAX_ISSUE_WIDTH];
    bool windowRegWrite[MAX_ISSUE_WIDTH];
    reg_t windowAluResult[MAX_ISSUE_WIDTH];                   // filled by execute_m()
    reg_t windowMemResult[MAX_ISSUE_WIDTH];                    // filled by read_m()

    // Branch Target Buffer (BTB) & 2-bit saturating counter
    struct BTBEntry {
        reg_t target_pc;
        uint8_t state; // 2-bit counter: 0=SNT, 1=WNT, 2=WT, 3=ST
        bool valid;
    };
    static constexpr int BTB_SIZE = 128;
    BTBEntry btb[BTB_SIZE];
    size_t btb_hash(reg_t addr) const { return (addr >> 2) & (BTB_SIZE - 1); }

    // Per-instruction decode/execute/read, used only by the superscalar path
    // (duplicated from decode()/execute()/read(), not shared, so the
    // original path is untouched).
    static InstructionFields decode_one(reg_t word);
    reg_t execute_one(const InstructionFields& f, reg_t slot_pc, reg_t& out_next_pc, bool& out_next_pc_set);
    void read_one(const InstructionFields& f, reg_t aluResult, bool slot_mem_read_enable, bool slot_mem_write_enable, reg_t& out_memResult);

    // Pure RV32IM ALU: opcodes 0x13 (OP-IMM), 0x33 (OP, incl. M ext), 0x37
    // (LUI), 0x17 (AUIPC). a/b are the already-read rs1/rs2 values, pc is
    // the instruction's own PC (only AUIPC uses it). The single source of
    // truth for these ops' semantics -- execute_one() below calls it, and
    // so does SIMTCore, once per lane.
    static reg_t alu_exec(const InstructionFields& f, reg_t a, reg_t b, reg_t pc);
    // BEQ/BNE/BLT/BGE/BLTU/BGEU condition evaluation (opcode 0x63), shared
    // the same way -- SIMTCore uses this for a SIMT-uniform branch, raising
    // if active lanes disagree (there's no per-lane PC to diverge to here).
    static bool branch_taken(uint8_t funct3, reg_t a, reg_t b);

    static bool is_branch(uint8_t opcode) { return opcode == 0x63; }
    static bool is_jump(uint8_t opcode) { return opcode == 0x6F || opcode == 0x67; }
    static bool is_branch_or_jump(uint8_t opcode);
    // Memory-port class for hazard_scan()'s structural-hazard checks 
    enum MemClass { MEM_NONE, MEM_BANK_A, MEM_BANK_B, MEM_STORE };
    static MemClass mem_class(uint8_t opcode, uint8_t funct3);
    // Which of rs1/rs2 a given opcode/funct3 actually reads (decode_one()
    // always populates both bit fields regardless of opcode).
    static void reg_usage(uint8_t opcode, uint8_t funct3, bool& uses_rs1, bool& uses_rs2);
    // LAUNCH (0x5B), shared by read() and read_m(): runs the grid, adds the
    // device cycles, halts if the GPU faulted or timed out.
    void run_launch(uint8_t funct3, reg_t entry, reg_t dims, reg_t launch_pc);
public:
    // Constructor
    CPU(Memory* mem_ptr);

    // Core execution pipeline methods
    void fetch();
    void decode();
    void execute();
    void read();
    void writeback();

    // In-order superscalar pipeline -- call all six in sequence each cycle
    // in place of the five methods above. See docs/superscalar.md.
    void fetch_n();
    void decode_all();
    void hazard_scan();
    void execute_m();
    void read_m();
    void writeback_m();
    void set_issue_width(int n); // clamped to [1, MAX_ISSUE_WIDTH]
    int issue_width() const { return issueWidth; }
    int last_issue_count() const { return issueCount; } // m from the most recent hazard_scan()
    // Per-slot opcode/funct3 for the most recent superscalar cycle (slot in
    // [0, last_issue_count())) -- e.g. for a harness annotating NPU
    // instructions in a trace.
    uint8_t issued_opcode(int slot) const { return windowDecoded[slot].opcode; }
    uint8_t issued_funct3(int slot) const { return windowDecoded[slot].funct3; }
    reg_t slot_pc(int slot) const { return windowSlotPC[slot]; }
    bool is_cancelled(int slot) const { return windowCancelled[slot]; }
    bool is_speculative(int slot) const { return windowIsSpeculative[slot]; }
    bool is_branch_slot(int slot) const { return windowIsBranch[slot]; }
    // Debugging method to print the CPU state
    void print_state();
    // True once execute()/execute_m() has seen a jump/branch that targets
    // its own address
    bool is_halted() const { return halted; }
    reg_t current_pc() const { return pc; }
    // Opcode/funct3 of the instruction most recently run through decode()
    // (the single-instruction path) -- e.g. for a harness annotating NPU
    // instructions in a trace.
    uint8_t last_opcode() const { return decodedInstruction.opcode; }
    uint8_t last_funct3() const { return decodedInstruction.funct3; }
    // Overrides the reset PC (default 0). For harnesses that load a binary at
    // a non-zero base address -- e.g. tests/riscv-arch-test/harness.cpp,
    // whose linker script uses a non-zero origin so qemu-riscv32 can also run
    // the same binary as its reference (qemu-user refuses to mmap address 0).
    void set_pc(reg_t addr) { pc = addr; }
    // Connects the GPU that LAUNCH (custom-2, opcode 0x5B) runs kernels on.
    // Without one, LAUNCH prints an error and does nothing.
    void attach_gpu(GridLauncher* g) { gpu = g; }
    uint64_t get_cycle_count() const { return cycle_count; }
    // Custom-2: host-side grid launch -- .insn r 0x5B, 0, 0, x0, rs1, rs2
    // rs1 = kernel entry, rs2 = (num_blocks << 16) | threads_per_block,
    // a0 (x10) = kernel argument (read implicitly). See docs/CUDA/grid_launch_plan.md 4.1.
    static constexpr uint8_t OPCODE_LAUNCH = 0x5B;
    // The amount of memory to reserve for the host stack.
    static constexpr uint32_t HOST_STACK_RESERVE = 0x10000; // 64 KB (Other devices can place it's stack below it)
};

#endif // CPU_H