#include "../include/CPU.h"
#include "../include/NPU.h"
#include "../include/GridLauncher.h"
#include <iostream>
#include <cstring>
#include <iomanip>

// Constructor: Initializes the CPU state
CPU::CPU(Memory* mem_ptr) {
    memory = mem_ptr;
    gpu = nullptr;

    std::memset(registers, 0, sizeof(registers));
    pc = 0;
    cycle_count = 0;
    halted = false;
    issueWidth = MAX_ISSUE_WIDTH;
    issueCount = 0;
    fetchCount = 0;
    fetchNextPC = 0;
    for (int i = 0; i < BTB_SIZE; i++) {
        btb[i].target_pc = 0;
        btb[i].state = 0;
        btb[i].valid = false;
    }
    // Block map reset: 64x64 grid, every cell pickable (same as blk.clr 64).
    blkDim = BLK_MAX_DIM;
    for (uint32_t r = 0; r < BLK_MAX_DIM; r++)
        blkValid[r] = ~0ull;
    // Bucket heads reset: every bucket empty (same as bkt.clr).
    for (uint32_t v = 0; v < BKT_BUCKETS; v++) {
        bktHead[v] = 0;
        bktValid[v] = false;
    }
}

void CPU::set_issue_width(int n) {
    if (n < 1) n = 1;
    if (n > MAX_ISSUE_WIDTH) n = MAX_ISSUE_WIDTH;
    issueWidth = n;
}

// Enforces that register x0 is always hardwired to 0
void CPU::enforce_zero_register() {
    registers[0] = 0;
}

// A handy debug function to see what is inside your registers
void CPU::print_state() {
    std::cout << "--- CPU State ---" << std::endl;
    std::cout << "PC: 0x" << std::hex << std::setfill('0') << std::setw(8) << pc << std::endl;
    std::cout << "Cycles: " << std::dec << cycle_count << std::endl;

    for (int i = 0; i < NUM_REGS; i++) {
        std::cout << "x" << std::dec << i << ": 0x" 
                  << std::hex << std::setfill('0') << std::setw(8) << registers[i] << "  ";
        if ((i + 1) % 4 == 0) std::cout << std::endl; // Print 4 registers per row
    }
    std::cout << "-----------------" << std::endl;
}

void CPU::fetch() {
    // Logic to read from memory using PC and store in instruction
    // update the PC to point to the next instruction
    // if it sees isBranchTaken, then it should update the PC to branchPC instead of incrementing by 4
    // also update the decode and execute methods
    instruction = memory->read_word(pc);
    next_pc = pc + 4; // Move to the next instruction (assuming 4-byte instructions)
    mem_read_enable = false; // Reset memory read enable
    mem_write_enable = false; // Reset memory write enable
    reg_write_enable = false; // Reset register write enable
}


void CPU::decode() {
    decodedInstruction.opcode = instruction & 0x7F; // Extract the last 7 bits for opcode
    decodedInstruction.rd = (instruction >> 7) & 0x1F; // bits 11-7
    decodedInstruction.rs1 = (instruction >> 15) & 0x1F; // bits 19-15
    decodedInstruction.rs2 = (instruction >> 20) & 0x1F; // bits 24-20
    decodedInstruction.funct3 = (instruction >> 12) & 0x07; // bits 14-12
    decodedInstruction.funct7 = (instruction >> 25) & 0x7F; // bits 31-25

    // 1. U-Type (Upper Immediate)
    // Used in: LUI, AUIPC
    decodedInstruction.imm_U = instruction & 0xFFFFF000;

    // 2. I-Type (Standard Immediate)
    // Used in: ADDI, SLTI, Loads (LW, LB, etc.), JALR
    // Sign-extends bits 31:20
    decodedInstruction.imm_I = (uint32_t)((int32_t)instruction >> 20);

    // 3. S-Type (Store Immediate)
    // Used in: SW, SH, SB
    // Combines bits 31:25 and 11:7, sign-extended
    decodedInstruction.imm_S = (uint32_t)(
        (((int32_t)instruction >> 20) & 0xFFFFFFE0) | 
        ((instruction >> 7) & 0x0000001F)
    );

    // 4. B-Type (Branch Immediate)
    // Used in: BEQ, BNE, BLT, BGE, BLTU, BGEU
    // Combines bits 31, 7, 30:25, and 11:8, sign-extended, bit 0 is 0
    decodedInstruction.imm_B = (uint32_t)(
        (((int32_t)instruction >> 19) & 0xFFFFF000) | 
        ((instruction << 4)  & 0x00000800) | 
        ((instruction >> 20) & 0x000007E0) | 
        ((instruction >> 7)  & 0x0000001E)
    );

    // 5. J-Type (Jump Immediate)
    // Used in: JAL
    // Combines bits 31, 19:12, 20, and 30:21, sign-extended, bit 0 is 0
    decodedInstruction.imm_J = (uint32_t)(
        (((int32_t)instruction >> 11) & 0xFFF00000) | 
        (instruction & 0x000FF000) |                  
        ((instruction >> 9)  & 0x00000800) |           
        ((instruction >> 20) & 0x000007FE)
    );

    // 6. Z-Type / Shamt (Zero-extended Immediate / Shift Amount)
    // Used in: CSRRWI, CSRRSI, CSRRCI, SLLI, SRLI, SRAI
    // Extracts bits 19:15, zero-extended
    uint32_t imm_Z = (instruction >> 15) & 0x0000001F;
    decodedInstruction.imm_Z = imm_Z;
    // --- CONTROL UNIT ---
    // Set pipeline control signals based on the opcode
    switch (decodedInstruction.opcode) {
        
        // ---------------------------------------------------------
        // Math & Logic Instructions (Result goes to Register)
        // ---------------------------------------------------------
        case 0x13: // I-type (ADDI, SLTI, etc.)
        case 0x33: // R-type (ADD, SUB, etc.)
        case 0x37: // LUI
        case 0x17: // AUIPC
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = true;  // Save ALU result to 'rd'
            break;

        // Custom-0: NPU tile transfer / memory print (.insn r 0x0B, funct3,
        // 0, x0, rs1, rs2) -- funct3 0/1 read a 16x16 tile stored
        // contiguously in memory (rs1=base, rs2 unused) into the NPU's data
        // window; funct3 2 writes the NPU's result tile back into a
        // row-major matrix in memory (rs1=base, rs2=row stride in bytes);
        // funct3 3 prints an NxN region of memory (rs1=base, rs2=N) via
        // Memory::print_matrix(); funct3 4 preloads C with a bias row
        // (rs1=base); funct3 5 stores C as int8 through the output stage
        // (rs1=base, rs2=shift). All done in read() (run_npu_op()),
        // touching neither a register nor the scalar mem_read/write path.
        case 0x0B:
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = false;
            break;

        // Custom-2: LAUNCH (.insn r 0x5B, 0, 0, x0, rs1, rs2) -- runs a kernel
        // grid on the attached GPU in read(); writes no register, and the
        // scalar mem path stays off (the GPU does its own memory accesses).
        case OPCODE_LAUNCH:
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = false;
            break;

        // Custom-3: token-selection unit (see CPU.h). The work is done in
        // read() (run_toksel()); blk/bkt.ins/bkt.head write rd, the rest don't.
        case OPCODE_TOKSEL:
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = toksel_writes_rd(decodedInstruction.funct3);
            break;

        // SYSTEM: only rdcycle/rdcycleh (see CPU.h), done in read()
        // (run_system()); they write rd, anything else writes nothing.
        case OPCODE_SYSTEM:
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = is_counter_read(decodedInstruction);
            break;

        // ---------------------------------------------------------
        // Memory Instructions
        // ---------------------------------------------------------
        case 0x03: // Loads (LW, LH, LB, etc.)
            mem_read_enable  = true;  // Read from RAM
            mem_write_enable = false;
            reg_write_enable = true;  // Save RAM data to 'rd'
            break;

        case 0x23: // Stores (SW, SH, SB)
            mem_read_enable  = false;
            mem_write_enable = true;  // Write to RAM
            reg_write_enable = false; // Does not update any registers
            break;

        // ---------------------------------------------------------
        // Control Flow (Branches & Jumps)
        // ---------------------------------------------------------
        case 0x63: // Branches (BEQ, BNE, BLT, etc.)
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = false; // Branches only update PC, not 'rd'
            break;

        case 0x6F: // JAL (Jump and Link)
        case 0x67: // JALR (Jump and Link Register)
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = true;  // Save Return Address (PC + 4) to 'rd'
            break;

        // ---------------------------------------------------------
        // Default Fallback (Safety)
        // ---------------------------------------------------------
        default:
            // For unknown/unimplemented instructions, default to NO-OP
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = false;
            break;
    }
}
void CPU::execute() {
    // Instructions are of two type - branch and non branch
    // For non branch instructions, it is handled by ALU
    // For branch instructions, it is handled by the branch unit
    switch (decodedInstruction.opcode) {
        //Integer Register-Immediate Instructions (I-type)
        // 0010011 ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI
        case 0x13: {
            //ADDI adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored  
            //the result is simply the low XLEN bits of the result. ADDI rd, rs1, 0 is used to implement the MV
            //rd, rs1 assembler pseudo-instruction.
            switch (decodedInstruction.funct3) {
                case 0x0: // ADDI
                    aluResult = registers[decodedInstruction.rs1] + decodedInstruction.imm_I;
                    break;
                case 0x2: // SLTI
                    aluResult = ((int32_t)registers[decodedInstruction.rs1] < (int32_t)decodedInstruction.imm_I) ? 1 : 0;
                    break;
                case 0x3: // SLTIU
                    aluResult = ((uint32_t)registers[decodedInstruction.rs1] < (uint32_t)decodedInstruction.imm_I) ? 1 : 0;
                    break;
                case 0x4: // XORI
                    aluResult = registers[decodedInstruction.rs1] ^ decodedInstruction.imm_I;
                    break;
                case 0x6: // ORI
                    aluResult = registers[decodedInstruction.rs1] | decodedInstruction.imm_I;
                    break;
                case 0x7: // ANDI
                    aluResult = registers[decodedInstruction.rs1] & decodedInstruction.imm_I;
                    break;
                case 0x1: // SLLI
                    aluResult = registers[decodedInstruction.rs1] << (decodedInstruction.imm_I & 0x1F);
                    break;
                case 0x5: // SRLI and SRAI
                    if ((decodedInstruction.imm_I >> 10) & 0x1) { // Check if it's SRAI
                        aluResult = (int32_t)registers[decodedInstruction.rs1] >> (decodedInstruction.imm_I & 0x1F); // Arithmetic right shift
                    } else { // SRLI
                        aluResult = registers[decodedInstruction.rs1] >> (decodedInstruction.imm_I & 0x1F); // Logical right shift
                    }
                    break;
            }
            break;
        }
        // LUI and AUIPC Instructions (U-type)
        case 0x37: { // LUI 1110111
            aluResult = decodedInstruction.imm_U;
            break;
        }
        case 0x17: { // AUIPC 0010111
            aluResult = pc + decodedInstruction.imm_U;
            break;
        }
        // Integer Register-Register Instructions (R-type)
        case 0x33: {
            switch (decodedInstruction.funct7){
                case 0x00:
                case 0x20:
                    switch (decodedInstruction.funct3) {
                        case 0x0: // ADD and SUB
                            if (decodedInstruction.funct7 == 0x00) { // ADD
                                aluResult = registers[decodedInstruction.rs1] + registers[decodedInstruction.rs2];
                            } else if (decodedInstruction.funct7 == 0x20) { // SUB
                                aluResult = registers[decodedInstruction.rs1] - registers[decodedInstruction.rs2];  
                            }
                            break;
                        case 0x1: // SLL
                            aluResult = registers[decodedInstruction.rs1] << (registers[decodedInstruction.rs2] & 0x1F);
                            break;
                        case 0x2: // SLT (signed comparison, unlike SLTU below)
                            aluResult = ((int32_t)registers[decodedInstruction.rs1] < (int32_t)registers[decodedInstruction.rs2]) ? 1 : 0;
                            break;
                        case 0x3: // SLTU
                            aluResult = ((uint32_t)registers[decodedInstruction.rs1] < (uint32_t)registers[decodedInstruction.rs2]) ? 1 : 0;
                            break;
                        case 0x4: // XOR
                            aluResult = registers[decodedInstruction.rs1] ^ registers[decodedInstruction.rs2];
                            break;
                        case 0x5: // SRL and SRA
                            if (decodedInstruction.funct7 == 0x00) { // SRL
                                aluResult = registers[decodedInstruction.rs1] >> (registers[decodedInstruction.rs2] & 0x1F);
                            } else if (decodedInstruction.funct7 == 0x20 ) { // SRA
                                aluResult = (int32_t)registers[decodedInstruction.rs1] >> (registers[decodedInstruction.rs2] & 0x1F); // Arithmetic right shift
                            }
                            break;
                        case 0x6: // OR
                            aluResult = registers[decodedInstruction.rs1] | registers[decodedInstruction.rs2];
                            break;
                        case 0x7: // AND
                            aluResult = registers[decodedInstruction.rs1] & registers[decodedInstruction.rs2];
                            break;  
                    }
                break;
                //Multiplication and Division Instructions (M-type)
                case 0x01: {
                    switch (decodedInstruction.funct3) {
                        case 0x0: // MUL
                            aluResult = registers[decodedInstruction.rs1] * registers[decodedInstruction.rs2];
                            break;
                        case 0x1: // MULH
                            aluResult = ((int64_t)(int32_t)registers[decodedInstruction.rs1] * (int64_t)(int32_t)registers[decodedInstruction.rs2]) >> 32;
                            break;
                        case 0x2: // MULHSU
                            aluResult = ((int64_t)(int32_t)registers[decodedInstruction.rs1] * (uint64_t)(uint32_t)registers[decodedInstruction.rs2]) >> 32;
                            break;
                        case 0x3: // MULHU
                            aluResult = ((uint64_t)(uint32_t)registers[decodedInstruction.rs1] * (uint64_t)(uint32_t)registers[decodedInstruction.rs2]) >> 32;
                            break;
                        case 0x4: // DIV
                            if (registers[decodedInstruction.rs2] == 0) {
                                aluResult = -1; // Division by zero returns -1
                            } else if (registers[decodedInstruction.rs1] == 0x80000000 && (int32_t)registers[decodedInstruction.rs2] == -1) {
                                aluResult = 0x80000000; // Signed overflow (INT_MIN / -1): result is INT_MIN, no trap
                            } else {
                                aluResult = (int32_t)registers[decodedInstruction.rs1] / (int32_t)registers[decodedInstruction.rs2];
                            }
                            break;
                        case 0x5: // DIVU
                            if (registers[decodedInstruction.rs2] == 0) {
                                aluResult = UINT32_MAX; // Division by zero returns max unsigned value
                            } else {
                                aluResult = registers[decodedInstruction.rs1] / registers[decodedInstruction.rs2];
                            }
                            break;
                        case 0x6: // REM
                            if (registers[decodedInstruction.rs2] == 0) {
                                aluResult = registers[decodedInstruction.rs1]; // Remainder by zero returns dividend
                            } else if (registers[decodedInstruction.rs1] == 0x80000000 && (int32_t)registers[decodedInstruction.rs2] == -1) {
                                aluResult = 0; // Signed overflow (INT_MIN / -1): remainder is 0, no trap
                            } else {
                                aluResult = (int32_t)registers[decodedInstruction.rs1] % (int32_t)registers[decodedInstruction.rs2];
                            }
                            break;
                        case 0x7: // REMU
                            if (registers[decodedInstruction.rs2] == 0) {
                                aluResult = registers[decodedInstruction.rs1]; // Remainder by zero returns dividend
                            } else {
                                aluResult = registers[decodedInstruction.rs1] % registers[decodedInstruction.rs2];
                            }
                        }
                    }
                }
            break;
        }
        // Load and Store Instructions (I-type for load, S-type for store)
        case 0x03: {// Load
            // Calculate address and store in our pipeline register
            aluResult = registers[decodedInstruction.rs1] + decodedInstruction.imm_I;
            break;
        }
        case 0x23: {// Store
            // Calculate address (Note: Stores usually use imm_S, not I)
            aluResult = registers[decodedInstruction.rs1] + decodedInstruction.imm_S; 
            break;
        }
        // Branch Instructions (B-type)
        case 0x63: { // Branch instructions
        bool branch_taken = false;
        
        // Grab register values for cleaner code

        // 1. Evaluate the condition
        switch (decodedInstruction.funct3) {
            case 0x0: // BEQ (Branch if Equal)
                branch_taken = (registers[decodedInstruction.rs1] == registers[decodedInstruction.rs2]);
                break;
            case 0x1: // BNE (Branch if Not Equal)
                branch_taken = (registers[decodedInstruction.rs1] != registers[decodedInstruction.rs2]);
                break;
            case 0x4: // BLT (Branch if Less Than, Signed)
                branch_taken = ((int32_t)registers[decodedInstruction.rs1] < (int32_t)registers[decodedInstruction.rs2]);
                break;
            case 0x5: // BGE (Branch if Greater or Equal, Signed)
                branch_taken = ((int32_t)registers[decodedInstruction.rs1] >= (int32_t)registers[decodedInstruction.rs2]);
                break;
            case 0x6: // BLTU (Branch if Less Than, Unsigned)
                branch_taken = (registers[decodedInstruction.rs1] < registers[decodedInstruction.rs2]);
                break;
            case 0x7: // BGEU (Branch if Greater or Equal, Unsigned)
                branch_taken = (registers[decodedInstruction.rs1] >= registers[decodedInstruction.rs2]);
                break;
        }
        if (branch_taken) {
            // If the branch is taken, calculate the new PC
            branchPC = pc + decodedInstruction.imm_B;
            next_pc = branchPC; // Update the PC to the branch target
        }
        break;
        }
        case 0x6F: { // JAL (Jump and Link)
            aluResult = pc + 4; // Store return address in rd
            next_pc = pc + decodedInstruction.imm_J; // Jump to target
            break;
        }
        case 0x67: { // JALR (Jump and Link Register)
            aluResult = pc + 4; // Store return address in rd
            next_pc = (registers[decodedInstruction.rs1] + decodedInstruction.imm_I) & ~1; // Jump to target, ensure LSB is 0
            break;
        }
        case 0x0B: { // Custom-0: NPU strided tile transfer / memory print
            aluResult = registers[decodedInstruction.rs1]; // base address, consumed by read()
            break;
        }
        case OPCODE_LAUNCH: { // Custom-2: grid launch
            aluResult = registers[decodedInstruction.rs1]; // kernel entry, consumed by read()
            break;
        }
        case OPCODE_TOKSEL: { // Custom-3: token-selection unit
            aluResult = registers[decodedInstruction.rs1]; // cell index, D or score, consumed by read()
            break;
        }
    }
    // A jump/branch whose target is its own address (e.g. start.s's `_end: j _end`)
    // is this codebase's halt idiom -- the program has finished and is spinning forever.
    halted = (next_pc == pc);   
}

void CPU::read() {
    // ---------------------------------------------------------
    // Load Instructions (Read from RAM)
    // ---------------------------------------------------------
    if (mem_read_enable) {
        switch (decodedInstruction.funct3) {
            case 0x0: // LB (Load Byte, Sign-extended)
                memResult = (int32_t)(int8_t)memory->read_byte(aluResult); 
                break;
            case 0x1: // LH (Load Halfword, Sign-extended)
                memResult = (int32_t)(int16_t)memory->read_halfword(aluResult); 
                break;
            case 0x2: // LW (Load Word)
                memResult = memory->read_word(aluResult);
                break;
            case 0x4: // LBU (Load Byte, Unsigned)
                memResult = memory->read_byte(aluResult); 
                break;
            case 0x5: // LHU (Load Halfword, Unsigned)
                memResult = memory->read_halfword(aluResult); 
                break;  
        }
    }

    // ---------------------------------------------------------
    // Store Instructions (Write to RAM)
    // ---------------------------------------------------------
    if (mem_write_enable) {
        // Write to memory at the address calculated in Stage 3 (aluResult)
        switch (decodedInstruction.funct3) {
            case 0x0: // SB (Store Byte)
                memory->write_byte(aluResult, registers[decodedInstruction.rs2] & 0xFF);
                break;
            case 0x1: // SH (Store Halfword)
                memory->write_halfword(aluResult, registers[decodedInstruction.rs2] & 0xFFFF);
                break;
            case 0x2: // SW (Store Word)
                memory->write_word(aluResult, registers[decodedInstruction.rs2]);
                break;
        }
    }

    // ---------------------------------------------------------
    // Custom-0 (opcode 0x0B): NPU tile transfer, output stage or memory
    // print -- see run_npu_op() below.
    // ---------------------------------------------------------
    if (decodedInstruction.opcode == 0x0B) {
        run_npu_op(decodedInstruction.funct3, aluResult, registers[decodedInstruction.rs2]);
    }

    // ---------------------------------------------------------
    // Custom-2 (opcode 0x5B) funct3 0: LAUNCH. The CPU stalls here until the
    // whole grid has run, then carries on at pc+4; the device cycles are
    // added to this CPU's cycle count. Other funct3 values are reserved.
    // ---------------------------------------------------------
    if (decodedInstruction.opcode == OPCODE_LAUNCH)
        run_launch(decodedInstruction.funct3, aluResult, registers[decodedInstruction.rs2], pc);

    // ---------------------------------------------------------
    // Custom-3 (opcode 0x7B): token-selection unit. Its result replaces
    // aluResult, which writeback() then commits to rd like any ALU result.
    // ---------------------------------------------------------
    if (decodedInstruction.opcode == OPCODE_TOKSEL)
        aluResult = run_toksel(decodedInstruction.funct3, aluResult, registers[decodedInstruction.rs2], pc);

    // ---------------------------------------------------------
    // SYSTEM (opcode 0x73): rdcycle/rdcycleh. cycle_count is only advanced
    // in writeback(), so here it still counts the cycles before this one.
    // ---------------------------------------------------------
    if (decodedInstruction.opcode == OPCODE_SYSTEM)
        aluResult = run_system(decodedInstruction, instruction, pc);
}

// Custom-0 (opcode 0x0B), shared by the scalar read() and the superscalar
// read_one(). base = rs1. Touches neither a register nor the scalar
// mem_read/write path.
//
// funct3 0/1 (loads) read the source tile as one contiguous block of
// int8 elements (MAX_DIM*MAX_DIM bytes, i.e. 64 words for a 16x16 tile)
// -- the caller is responsible for laying tiles out contiguously in
// memory ahead of time, so this is a single access rather than 16
// separate strided row reads. rs2 is unused here.
//
// funct3 2 (store) still writes strided, since the destination is a
// plain row-major matrix (rows are BIG_DIM apart, not tile-contiguous):
// rs2 = row stride in bytes. C stays int32 (the accumulator), unlike the
// int8 A/B loads above.
//
// funct3 3 prints an NxN region (rs2 = N) via Memory::print_matrix,
// unrelated to the NPU -- just a general "print memory" instruction.
//
// funct3 4 (load bias) reads MAX_DIM contiguous int32 bias values and
// preloads C with them, bias[j] down every row of column j -- a tile then
// starts from its bias instead of RESET's zero. rs2 is unused.
//
// funct3 5 (store C as int8) runs the NPU's output stage (rounding shift,
// clamp, zero outside M x N) with shift = rs2 and writes the int8
// tile contiguously at base (TILE_ELEMS bytes, tile-major).
//
// funct3 6 (load B transposed) reads the same contiguous int8 tile as
// funct3 1 but writes it into B transposed, B[i][j] = src[j][i], for
// A x B^T (CRUDA/docs/design/gemm_bt.md). rs2 is unused.
void CPU::run_npu_op(uint8_t funct3, reg_t base, reg_t rs2_val) {
    switch (funct3) {
        case 0x0: // load Matrix A: memory -> NPU (tile is contiguous in memory)
            for (uint32_t i = 0; i < NPU::TILE_ELEMS / 4; i++) // int8: 4 elements per word
                memory->write_word(NPU::MAT_A_ADDR + i * 4, memory->read_word(base + i * 4));
            break;
        case 0x1: // load Matrix B: memory -> NPU (tile is contiguous in memory)
            for (uint32_t i = 0; i < NPU::TILE_ELEMS / 4; i++) // int8: 4 elements per word
                memory->write_word(NPU::MAT_B_ADDR + i * 4, memory->read_word(base + i * 4));
            break;
        case 0x2: // store Matrix C: NPU -> memory (rs2_val = row stride)
            for (uint32_t row = 0; row < NPU::MAX_DIM; row++)
                for (uint32_t col = 0; col < NPU::MAX_DIM; col++)
                    memory->write_word(base + row * rs2_val + col * 4,
                                        memory->read_word(NPU::RESULT_ADDR + (row * NPU::MAX_DIM + col) * 4));
            break;
        case 0x3: // print NxN region of memory at base (rs2_val = N)
            memory->print_matrix(base, rs2_val, rs2_val);
            break;
        case 0x4: // load bias: memory -> C, broadcast down each column (MAX_DIM int32)
            for (uint32_t j = 0; j < NPU::MAX_DIM; j++)
                memory->write_word(NPU::BIAS_ADDR + j * 4, memory->read_word(base + j * 4));
            break;
        case 0x5: // store C as int8 through the output stage (rs2_val = shift)
            memory->write_word(NPU::OUT_SHIFT_ADDR, rs2_val);
            for (uint32_t i = 0; i < NPU::TILE_ELEMS / 4; i++) // int8: 4 elements per word
                memory->write_word(base + i * 4, memory->read_word(NPU::OUT8_ADDR + i * 4));
            break;
        case 0x6: // load Matrix B transposed: B[i][j] = src[j][i] (tile is contiguous in memory)
            for (uint32_t i = 0; i < NPU::MAX_DIM; i++) {
                for (uint32_t j = 0; j < NPU::MAX_DIM; j += 4) { // int8: 4 elements per word
                    uint32_t word = 0;
                    for (uint32_t k = 0; k < 4; k++)
                        word |= (uint32_t)memory->read_byte(base + (j + k) * NPU::MAX_DIM + i) << (8 * k);
                    memory->write_word(NPU::MAT_B_ADDR + i * NPU::MAX_DIM + j, word);
                }
            }
            break;
    }
}

// LAUNCH, shared by the scalar read() and the superscalar read_m(). The CPU
// stalls here until the whole grid has run (it simply doesn't return before
// that), then carries on at pc+4 with the device cycles added to its own.
// entry = rs1, dims = rs2 = (num_blocks << 16) | threads_per_block, and the
// kernel argument is a0. Other funct3 values are reserved.
void CPU::run_launch(uint8_t funct3, reg_t entry, reg_t dims, reg_t launch_pc) {
    if (funct3 != 0) {
        std::cerr << "[CPU Error] reserved LAUNCH funct3 " << (int)funct3
                  << " at pc 0x" << std::hex << launch_pc << std::dec << " -- ignored\n";
        return;
    }
    if (gpu == nullptr) {
        std::cerr << "[CPU Error] LAUNCH at pc 0x" << std::hex << launch_pc << std::dec
                  << " with no GPU attached -- ignored\n";
        return;
    }
    uint32_t num_blocks        = dims >> 16;
    uint32_t threads_per_block = dims & 0xFFFF;
    cycle_count += gpu->launch_grid(entry, threads_per_block, num_blocks, registers[10]);
    // The GPU raised a flag (an instruction it can't run, or a runaway
    // kernel): the results are wrong from here on, so the host stops.
    if (gpu->faulted() || gpu->timed_out()) {
        std::cerr << "[CPU Error] LAUNCH at pc 0x" << std::hex << launch_pc << std::dec
                  << " failed on the GPU -- halting\n";
        halted = true;
    }
}

// SYSTEM, shared by the scalar read() and the superscalar read_m(). The CSR
// number is the I-immediate's 12 bits. rs1 must be x0: csrrs with any other
// rs1 also writes the CSR, and the counters are read-only.
bool CPU::is_counter_read(const InstructionFields& f) {
    uint32_t csr = (uint32_t)f.imm_I & 0xFFF;
    return f.opcode == OPCODE_SYSTEM && f.funct3 == 0x2 && f.rs1 == 0 &&
           (csr == 0xC00 || csr == 0xC80);
}

reg_t CPU::run_system(const InstructionFields& f, reg_t word, reg_t op_pc) {
    if (is_counter_read(f)) {
        uint32_t csr = (uint32_t)f.imm_I & 0xFFF;
        return (csr == 0xC00) ? (reg_t)cycle_count : (reg_t)(cycle_count >> 32);
    }
    std::cerr << "[CPU Error] unsupported SYSTEM instruction 0x" << std::hex << std::setfill('0')
              << std::setw(8) << word << std::setfill(' ') << " at pc 0x" << op_pc << std::dec
              << " -- ignored (only rdcycle/rdcycleh are implemented)\n";
    return 0;
}

// Custom-3, shared by the scalar read() and the superscalar read_m():
// funct3 0-2 go to the block map, 3-7 to the bucket heads.
reg_t CPU::run_toksel(uint8_t funct3, reg_t rs1_val, reg_t rs2_val, reg_t op_pc) {
    if (funct3 <= 0x2)
        return run_blk(funct3, rs1_val, op_pc);
    return run_bkt(funct3, rs1_val, rs2_val, op_pc);
}

// BLK block map.
// funct3 0 (blk): rs1_val = cell index D*r + c (upper bits ignored). Returns 1
// and clears the 3x3 neighbourhood if the cell is pickable, else returns 0
// and changes nothing. In hardware the neighbourhood is one-hot(r) and
// one-hot(c), each OR-ed with itself shifted by +-1; bits shifted past the
// grid edge drop off, so clipping needs no bounds checks.
// funct3 1 (blk.clr): rs1_val = D (32 or 64); every cell pickable again.
reg_t CPU::run_blk(uint8_t funct3, reg_t rs1_val, reg_t op_pc) {
    uint64_t row_mask = (blkDim == 64) ? ~0ull : ((1ull << blkDim) - 1); // columns 0..D-1
    switch (funct3) {
        case 0x0: { // blk rd, rs1: test-and-claim
            uint32_t col_bits = (blkDim == 64) ? 6 : 5;
            uint32_t c = rs1_val & (blkDim - 1);
            uint32_t r = (rs1_val >> col_bits) & (blkDim - 1);
            if (((blkValid[r] >> c) & 1) == 0)
                return 0;
            uint64_t cell = 1ull << c;
            uint64_t cols = (cell | (cell << 1) | (cell >> 1)) & row_mask;
            if (r > 0)
                blkValid[r - 1] &= ~cols;
            blkValid[r] &= ~cols;
            if (r + 1 < blkDim)
                blkValid[r + 1] &= ~cols;
            return 1;
        }
        case 0x1: { // blk.clr rs1: rs1 = D
            if (rs1_val != 32 && rs1_val != 64) {
                std::cerr << "[CPU Error] blk.clr with grid width " << rs1_val
                          << " (must be 32 or 64) at pc 0x" << std::hex << op_pc << std::dec
                          << " -- ignored\n";
                return 0;
            }
            blkDim = rs1_val;
            row_mask = (blkDim == 64) ? ~0ull : ((1ull << blkDim) - 1);
            for (uint32_t row = 0; row < BLK_MAX_DIM; row++)
                blkValid[row] = (row < blkDim) ? row_mask : 0;
            return 0;
        }
        default:
            std::cerr << "[CPU Error] reserved BLK funct3 " << (int)funct3
                      << " at pc 0x" << std::hex << op_pc << std::dec << " -- ignored\n";
            return 0;
    }
}

// BKT bucket heads: one linked-list head per 10-bit score value. In hardware
// bktHead is a 1024 x 12-bit SRAM and bktValid 1024 flops; "empty" (-1) is
// never stored, it is what a read returns while the valid bit is 0.
// funct3 3 (bkt.clr): every bucket empty -- resets the valid bits only.
// funct3 4 (bkt.ins): rs1_val = cell index (bits 11:0), rs2_val = score (bits
//   9:0). Returns the old head (-1 if the bucket was empty), then the cell
//   becomes the new head. The caller stores the old head as next[cell].
// funct3 5 (bkt.head): rs1_val = score (bits 9:0). Returns the head, -1 if empty.
reg_t CPU::run_bkt(uint8_t funct3, reg_t rs1_val, reg_t rs2_val, reg_t op_pc) {
    switch (funct3) {
        case 0x3: // bkt.clr
            for (uint32_t v = 0; v < BKT_BUCKETS; v++)
                bktValid[v] = false;
            return 0;
        case 0x4: { // bkt.ins rd, rs1, rs2
            uint32_t v = rs2_val & (BKT_BUCKETS - 1);
            reg_t old_head = bktValid[v] ? (reg_t)bktHead[v] : (reg_t)-1;
            bktHead[v] = rs1_val & 0xFFF;
            bktValid[v] = true;
            return old_head;
        }
        case 0x5: { // bkt.head rd, rs1
            uint32_t v = rs1_val & (BKT_BUCKETS - 1);
            return bktValid[v] ? (reg_t)bktHead[v] : (reg_t)-1;
        }
        default:
            std::cerr << "[CPU Error] reserved BKT funct3 " << (int)funct3
                      << " at pc 0x" << std::hex << op_pc << std::dec << " -- ignored\n";
            return 0;
    }
}
void CPU::writeback() {
    // Logic to write results back to registers
    if (reg_write_enable) {
        if (decodedInstruction.rd){
            if (mem_read_enable) {
                registers[decodedInstruction.rd] = memResult; // Write memory result to register
            } else {
                registers[decodedInstruction.rd] = aluResult; // Write ALU result to register
            }
        }
    }
    // Always enforce x0=0 at the end of execution
    enforce_zero_register();
    //update the pc
    pc = next_pc;
    cycle_count++;
}

// =============================================================================
// In-order superscalar issue -- see docs/superscalar.md for the design.
// =============================================================================

CPU::InstructionFields CPU::decode_one(reg_t word) {
    InstructionFields f;
    f.opcode = word & 0x7F;
    f.rd = (word >> 7) & 0x1F;
    f.rs1 = (word >> 15) & 0x1F;
    f.rs2 = (word >> 20) & 0x1F;
    f.funct3 = (word >> 12) & 0x07;
    f.funct7 = (word >> 25) & 0x7F;
    f.imm_U = word & 0xFFFFF000;
    f.imm_I = (uint32_t)((int32_t)word >> 20);
    f.imm_S = (uint32_t)(
        (((int32_t)word >> 20) & 0xFFFFFFE0) |
        ((word >> 7) & 0x0000001F)
    );
    f.imm_B = (uint32_t)(
        (((int32_t)word >> 19) & 0xFFFFF000) |
        ((word << 4)  & 0x00000800) |
        ((word >> 20) & 0x000007E0) |
        ((word >> 7)  & 0x0000001E)
    );
    f.imm_J = (uint32_t)(
        (((int32_t)word >> 11) & 0xFFF00000) |
        (word & 0x000FF000) |
        ((word >> 9)  & 0x00000800) |
        ((word >> 20) & 0x000007FE)
    );
    f.imm_Z = (word >> 15) & 0x0000001F;
    return f;
}

reg_t CPU::alu_exec(const InstructionFields& f, reg_t a, reg_t b, reg_t pc) {
    reg_t result = 0;
    switch (f.opcode) {
        case 0x13: {
            switch (f.funct3) {
                case 0x0: // ADDI
                    result = a + f.imm_I;
                    break;
                case 0x2: // SLTI
                    result = ((int32_t)a < (int32_t)f.imm_I) ? 1 : 0;
                    break;
                case 0x3: // SLTIU
                    result = ((uint32_t)a < (uint32_t)f.imm_I) ? 1 : 0;
                    break;
                case 0x4: // XORI
                    result = a ^ f.imm_I;
                    break;
                case 0x6: // ORI
                    result = a | f.imm_I;
                    break;
                case 0x7: // ANDI
                    result = a & f.imm_I;
                    break;
                case 0x1: // SLLI
                    result = a << (f.imm_I & 0x1F);
                    break;
                case 0x5: // SRLI and SRAI
                    if ((f.imm_I >> 10) & 0x1) {
                        result = (int32_t)a >> (f.imm_I & 0x1F);
                    } else {
                        result = a >> (f.imm_I & 0x1F);
                    }
                    break;
            }
            break;
        }
        case 0x37: { // LUI
            result = f.imm_U;
            break;
        }
        case 0x17: { // AUIPC
            result = pc + f.imm_U;
            break;
        }
        case 0x33: {
            switch (f.funct7) {
                case 0x00:
                case 0x20:
                    switch (f.funct3) {
                        case 0x0: // ADD and SUB
                            if (f.funct7 == 0x00) {
                                result = a + b;
                            } else if (f.funct7 == 0x20) {
                                result = a - b;
                            }
                            break;
                        case 0x1: // SLL
                            result = a << (b & 0x1F);
                            break;
                        case 0x2: // SLT
                            result = ((int32_t)a < (int32_t)b) ? 1 : 0;
                            break;
                        case 0x3: // SLTU
                            result = ((uint32_t)a < (uint32_t)b) ? 1 : 0;
                            break;
                        case 0x4: // XOR
                            result = a ^ b;
                            break;
                        case 0x5: // SRL and SRA
                            if (f.funct7 == 0x00) {
                                result = a >> (b & 0x1F);
                            } else if (f.funct7 == 0x20) {
                                result = (int32_t)a >> (b & 0x1F);
                            }
                            break;
                        case 0x6: // OR
                            result = a | b;
                            break;
                        case 0x7: // AND
                            result = a & b;
                            break;
                    }
                    break;
                case 0x01: {
                    switch (f.funct3) {
                        case 0x0: // MUL
                            result = a * b;
                            break;
                        case 0x1: // MULH
                            result = ((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32;
                            break;
                        case 0x2: // MULHSU
                            result = ((int64_t)(int32_t)a * (uint64_t)(uint32_t)b) >> 32;
                            break;
                        case 0x3: // MULHU
                            result = ((uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b) >> 32;
                            break;
                        case 0x4: // DIV
                            if (b == 0) {
                                result = -1;
                            } else if (a == 0x80000000 && (int32_t)b == -1) {
                                result = 0x80000000;
                            } else {
                                result = (int32_t)a / (int32_t)b;
                            }
                            break;
                        case 0x5: // DIVU
                            if (b == 0) {
                                result = UINT32_MAX;
                            } else {
                                result = a / b;
                            }
                            break;
                        case 0x6: // REM
                            if (b == 0) {
                                result = a;
                            } else if (a == 0x80000000 && (int32_t)b == -1) {
                                result = 0;
                            } else {
                                result = (int32_t)a % (int32_t)b;
                            }
                            break;
                        case 0x7: // REMU
                            if (b == 0) {
                                result = a;
                            } else {
                                result = a % b;
                            }
                    }
                }
            }
            break;
        }
    }
    return result;
}

bool CPU::branch_taken(uint8_t funct3, reg_t a, reg_t b) {
    switch (funct3) {
        case 0x0: return a == b;                    // BEQ
        case 0x1: return a != b;                     // BNE
        case 0x4: return (int32_t)a < (int32_t)b;    // BLT
        case 0x5: return (int32_t)a >= (int32_t)b;   // BGE
        case 0x6: return a < b;                       // BLTU
        case 0x7: return a >= b;                       // BGEU
    }
    return false;
}

reg_t CPU::execute_one(const InstructionFields& f, reg_t slot_pc, reg_t& out_next_pc, bool& out_next_pc_set) {
    reg_t result = 0;
    out_next_pc_set = false;
    switch (f.opcode) {
        case 0x13: case 0x37: case 0x17: case 0x33: // OP-IMM / LUI / AUIPC / OP (incl. M ext)
            result = alu_exec(f, registers[f.rs1], registers[f.rs2], slot_pc);
            break;
        case 0x03: { // Load address calc
            result = registers[f.rs1] + f.imm_I;
            break;
        }
        case 0x23: { // Store address calc
            result = registers[f.rs1] + f.imm_S;
            break;
        }
        case 0x63: { // Branch
            if (branch_taken(f.funct3, registers[f.rs1], registers[f.rs2])) {
                out_next_pc = slot_pc + f.imm_B;
                out_next_pc_set = true;
            }
            break;
        }
        case 0x6F: { // JAL
            result = slot_pc + 4;
            out_next_pc = slot_pc + f.imm_J;
            out_next_pc_set = true;
            break;
        }
        case 0x67: { // JALR
            result = slot_pc + 4;
            out_next_pc = (registers[f.rs1] + f.imm_I) & ~1;
            out_next_pc_set = true;
            break;
        }
        case OPCODE_LAUNCH: { // Custom-2: grid launch
            result = registers[f.rs1]; // kernel entry, consumed by read_m()
            break;
        }
        case 0x0B: { // Custom-0: NPU tile transfer / memory print
            result = registers[f.rs1]; // base address, consumed by read_one()
            break;
        }
        case OPCODE_TOKSEL: { // Custom-3: token-selection unit
            result = registers[f.rs1]; // cell index, D or score, consumed by read_m()
            break;
        }
    }
    return result;
}

void CPU::read_one(const InstructionFields& f, reg_t aluResult, bool slot_mem_read_enable, bool slot_mem_write_enable, reg_t& out_memResult) {
    if (slot_mem_read_enable) {
        switch (f.funct3) {
            case 0x0: out_memResult = (int32_t)(int8_t)memory->read_byte(aluResult); break;
            case 0x1: out_memResult = (int32_t)(int16_t)memory->read_halfword(aluResult); break;
            case 0x2: out_memResult = memory->read_word(aluResult); break;
            case 0x4: out_memResult = memory->read_byte(aluResult); break;
            case 0x5: out_memResult = memory->read_halfword(aluResult); break;
        }
    }

    if (slot_mem_write_enable) {
        switch (f.funct3) {
            case 0x0: memory->write_byte(aluResult, registers[f.rs2] & 0xFF); break;
            case 0x1: memory->write_halfword(aluResult, registers[f.rs2] & 0xFFFF); break;
            case 0x2: memory->write_word(aluResult, registers[f.rs2]); break;
        }
    }

    if (f.opcode == 0x0B) {
        run_npu_op(f.funct3, aluResult, registers[f.rs2]);
    }
}

bool CPU::is_branch_or_jump(uint8_t opcode) {
    return opcode == 0x63 || opcode == 0x6F || opcode == 0x67;
}

CPU::MemClass CPU::mem_class(uint8_t opcode, uint8_t funct3) {
    if (opcode == 0x23) return MEM_STORE; // scalar SW/SH/SB
    if (opcode == 0x0B) {
        switch (funct3) {
            case 0x0: return MEM_BANK_A; // NPU load A
            case 0x1: return MEM_BANK_B; // NPU load B
            case 0x2: return MEM_STORE;  // NPU store C -- treated like any other store
            case 0x4: return MEM_BANK_C;  // NPU load bias -- writes C, its own bank like A/B
            case 0x5: return MEM_STORE;  // NPU store C as int8 -- same write port as store C
            case 0x6: return MEM_BANK_B; // NPU load B transposed -- writes B, like load B
            default:  return MEM_NONE;   // funct3 3: debug memory print, never a hazard
        }
    }
    // Two BLK (or two BKT) instructions depend on each other through hidden
    // state (a hazard the rs1/rs2 RAW check can't see), so at most one of
    // each per window. The block map and the bucket heads are separate
    // storage, so one BLK and one BKT may share a window.
    if (opcode == OPCODE_TOKSEL) return (funct3 <= 0x2) ? MEM_BLK : MEM_BKT;
    return MEM_NONE;
}

void CPU::reg_usage(uint8_t opcode, uint8_t funct3, bool& uses_rs1, bool& uses_rs2) {
    switch (opcode) {
        case 0x13: // I-type ALU
        case 0x03: // Loads
        case 0x67: // JALR
            uses_rs1 = true;  uses_rs2 = false; break;
        case 0x33: // R-type ALU / M-extension
        case 0x23: // Stores
        case 0x63: // Branches
            uses_rs1 = true;  uses_rs2 = true;  break;
        case 0x37: // LUI
        case 0x17: // AUIPC
        case 0x6F: // JAL
            uses_rs1 = false; uses_rs2 = false; break;
        case OPCODE_LAUNCH: // Custom-2: rs1 = kernel entry, rs2 = dims (a0 is read too, see hazard_scan)
            uses_rs1 = true;  uses_rs2 = true;  break;
        case 0x0B: // Custom-0: rs1 is always the base address; rs2 is only
                   // meaningful for store-C (stride), memory-print (N) and
                   // store-C-int8 (shift).
            uses_rs1 = true;
            uses_rs2 = (funct3 == 0x2 || funct3 == 0x3 || funct3 == 0x5);
            break;
        case OPCODE_TOKSEL: // Custom-3: rs1 = cell index, D or score; rs2 = score (bkt.ins only)
            uses_rs1 = true;  uses_rs2 = (funct3 == 0x4); break;
        default:
            uses_rs1 = false; uses_rs2 = false; break;
    }
}

void CPU::fetch_n() {
    windowBasePC = pc;
    reg_t fetch_pc = pc;
    bool branch_seen_in_packet = false;
    fetchCount = 0;

    for (int slot = 0; slot < issueWidth; slot++) {
        reg_t instruction_word = memory->read_word(fetch_pc);
        windowWords[slot] = instruction_word;
        windowSlotPC[slot] = fetch_pc;
        windowIsSpeculative[slot] = branch_seen_in_packet;
        windowCancelled[slot] = false;

        uint8_t opcode = instruction_word & 0x7F;

        if (is_branch(opcode)) {
            windowIsBranch[slot] = true;
            size_t idx = btb_hash(fetch_pc);
            const BTBEntry& btb_entry = btb[idx];

            if (btb_entry.valid && btb_entry.state >= 2) {
                windowPredictedTaken[slot] = true;
                windowPredictedTarget[slot] = btb_entry.target_pc;
                fetch_pc = btb_entry.target_pc;
            } else {
                windowPredictedTaken[slot] = false;
                windowPredictedTarget[slot] = fetch_pc + 4;
                fetch_pc = fetch_pc + 4;
            }
            branch_seen_in_packet = true;
        } else if (is_jump(opcode)) {
            // No speculation across jumps: record slot and stop fetching further slots this cycle
            windowIsBranch[slot] = false;
            windowPredictedTaken[slot] = false;
            windowPredictedTarget[slot] = 0;
            fetch_pc = fetch_pc + 4;
            fetchCount = slot + 1;
            break;
        } else {
            windowIsBranch[slot] = false;
            windowPredictedTaken[slot] = false;
            windowPredictedTarget[slot] = 0;
            fetch_pc = fetch_pc + 4;
        }

        fetchCount = slot + 1;
    }

    fetchNextPC = fetch_pc;
}

void CPU::decode_all() {
    for (int i = 0; i < fetchCount; i++) {
        windowDecoded[i] = decode_one(windowWords[i]);
        switch (windowDecoded[i].opcode) {
            case 0x13: case 0x33: case 0x37: case 0x17: // ALU / LUI / AUIPC
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = true;
                break;
            case 0x0B: // Custom-0: NPU tile transfer / memory print
            case OPCODE_LAUNCH: // Custom-2: grid launch -- done in read_m(), writes no register
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = false;
                break;
            case OPCODE_TOKSEL: // Custom-3: done in read_m(); blk/bkt.ins/bkt.head write rd
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = toksel_writes_rd(windowDecoded[i].funct3);
                break;
            case OPCODE_SYSTEM: // done in read_m(); only rdcycle/rdcycleh write rd
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = is_counter_read(windowDecoded[i]);
                break;
            case 0x03: // Loads
                windowMemRead[i]  = true;
                windowMemWrite[i] = false;
                windowRegWrite[i] = true;
                break;
            case 0x23: // Stores
                windowMemRead[i]  = false;
                windowMemWrite[i] = true;
                windowRegWrite[i] = false;
                break;
            case 0x63: // Branches
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = false;
                break;
            case 0x6F: case 0x67: // JAL / JALR
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = true;
                break;
            default:
                windowMemRead[i]  = false;
                windowMemWrite[i] = false;
                windowRegWrite[i] = false;
                break;
        }
    }
}

// Scans the decoded window left to right, picking the largest prefix m that's
// safe to issue together this cycle (docs/superscalar.md has the full rules).
void CPU::hazard_scan() {
    issueCount = fetchCount;
    for (int i = 0; i < fetchCount; i++) {
        const InstructionFields& fi = windowDecoded[i];
        MemClass my_class = mem_class(fi.opcode, fi.funct3);

        // LAUNCH issues alone: it reads a0 implicitly (a RAW hazard the
        // rs1/rs2 check below can't see), it runs a whole grid that reads and
        // writes memory (a full barrier), and it can't be undone, so it must
        // never sit in a speculative slot -- slot 0 never is. The CPU stalls
        // in that cycle while the GPU runs.
        if (fi.opcode == OPCODE_LAUNCH) { issueCount = (i > 0) ? i : 1; return; }

        if (i > 0) {
            bool uses_rs1, uses_rs2;
            reg_usage(fi.opcode, fi.funct3, uses_rs1, uses_rs2);
            bool raw_hazard = false;
            for (int j = 0; j < i; j++) {
                if (!windowRegWrite[j] || windowDecoded[j].rd == 0) continue;
                uint8_t written_rd = windowDecoded[j].rd;
                if ((uses_rs1 && fi.rs1 == written_rd) || (uses_rs2 && fi.rs2 == written_rd)) {
                    raw_hazard = true;
                    break;
                }
            }
            if (raw_hazard) { issueCount = i; return; }

            // Memory-port conflicts against every earlier slot: two
            // instructions of the SAME MemClass conflict -- two stores
            // (single write port), or two loads into the same NPU bank
            // (that bank's single write port). A store and an NPU bank load
            // are DIFFERENT classes and no longer conflict with each other:
            // a bank load reads external memory and writes into the NPU's
            // own internal SRAM, a store writes external memory -- separate
            // read/write ports, so they don't contend for anything. Loads
            // into DIFFERENT banks, or anything with MemClass::NONE
            // (including non-memory instructions like ADDI/ADD), never
            // conflict with anything.
            if (my_class != MEM_NONE) {
                bool mem_conflict = false;
                for (int j = 0; j < i; j++) {
                    MemClass other_class = mem_class(windowDecoded[j].opcode, windowDecoded[j].funct3);
                    if (other_class == my_class) { mem_conflict = true; break; }
                }
                if (mem_conflict) { issueCount = i; return; }
            }
        }

        // Only non-speculated jumps truncate the window; branches do not truncate.
        if (is_jump(fi.opcode)) { issueCount = i + 1; return; }
    }
}

void CPU::execute_m() {
    bool misprediction_found = false;
    reg_t recovery_pc = 0;
    halted = false;

    for (int i = 0; i < issueCount; i++) {
        windowCancelled[i] = false;
    }

    for (int i = 0; i < issueCount; i++) {
        reg_t slot_pc = windowSlotPC[i];
        const InstructionFields& fi = windowDecoded[i];

        if (fi.opcode == 0x63) { // Branch
            bool actual_taken = false;
            switch (fi.funct3) {
                case 0x0: actual_taken = (registers[fi.rs1] == registers[fi.rs2]); break; // BEQ
                case 0x1: actual_taken = (registers[fi.rs1] != registers[fi.rs2]); break; // BNE
                case 0x4: actual_taken = ((int32_t)registers[fi.rs1] < (int32_t)registers[fi.rs2]); break; // BLT
                case 0x5: actual_taken = ((int32_t)registers[fi.rs1] >= (int32_t)registers[fi.rs2]); break; // BGE
                case 0x6: actual_taken = (registers[fi.rs1] < registers[fi.rs2]); break; // BLTU
                case 0x7: actual_taken = (registers[fi.rs1] >= registers[fi.rs2]); break; // BGEU
            }
            reg_t actual_target = slot_pc + fi.imm_B;
            reg_t correct_path = actual_taken ? actual_target : (slot_pc + 4);

            bool mispredicted = (actual_taken != windowPredictedTaken[i]) ||
                               (actual_taken && (windowPredictedTarget[i] != actual_target));

            // Update Predictor Table
            size_t btb_idx = btb_hash(slot_pc);
            btb[btb_idx].valid = true;
            btb[btb_idx].target_pc = actual_target;
            if (actual_taken) {
                if (btb[btb_idx].state < 3) btb[btb_idx].state++;
            } else {
                if (btb[btb_idx].state > 0) btb[btb_idx].state--;
            }

            if (actual_taken && actual_target == slot_pc) {
                halted = true;
            }

            if (mispredicted) {
                misprediction_found = true;
                recovery_pc = correct_path;

                // Pipeline Squash on Mispredict: squash only instructions issued after this branch in the current window
                for (int tail = i + 1; tail < issueCount; tail++) {
                    windowCancelled[tail] = true;
                    windowRegWrite[tail] = false;
                    windowMemWrite[tail] = false;
                    windowMemRead[tail] = false;
                }
                break;
            }
        } else {
            reg_t out_next_pc;
            bool out_next_pc_set = false;
            windowAluResult[i] = execute_one(fi, slot_pc, out_next_pc, out_next_pc_set);
            if (out_next_pc_set) {
                // Jump (JAL / JALR)
                windowNextPC = out_next_pc;
                if (out_next_pc == slot_pc) halted = true; // same halt idiom as execute()
            }
        }
    }

    if (misprediction_found) {
        windowNextPC = recovery_pc;
    } else if (!halted) {
        // If a jump did not set windowNextPC, advance to the first unissued instruction
        // or to fetchNextPC if all fetched instructions were issued.
        bool ended_with_jump = (issueCount > 0 && is_jump(windowDecoded[issueCount - 1].opcode));
        if (!ended_with_jump) {
            if (issueCount < fetchCount) {
                windowNextPC = windowSlotPC[issueCount];
            } else {
                windowNextPC = fetchNextPC;
            }
        }
    }
}

void CPU::read_m() {
    for (int i = 0; i < issueCount; i++) {
        if (windowCancelled[i]) continue;
        // LAUNCH always issues alone (see hazard_scan), so every earlier
        // instruction -- including whatever set a0 -- has already committed.
        if (windowDecoded[i].opcode == OPCODE_LAUNCH) {
            run_launch(windowDecoded[i].funct3, windowAluResult[i], registers[windowDecoded[i].rs2], windowSlotPC[i]);
            continue;
        }
        // Custom-3: a squashed (mispredicted) slot was skipped above, so it
        // never touches the block map or the bucket heads. The result replaces
        // the slot's ALU result, which writeback_m() commits to rd. rs2 is read
        // here, like NPU/LAUNCH: the RAW check guarantees no earlier slot in
        // this window writes it.
        if (windowDecoded[i].opcode == OPCODE_TOKSEL) {
            windowAluResult[i] = run_toksel(windowDecoded[i].funct3, windowAluResult[i],
                                            registers[windowDecoded[i].rs2], windowSlotPC[i]);
            continue;
        }
        // SYSTEM: every slot of the group reads the same count, the cycles
        // before this group (cycle_count only advances in writeback_m()).
        // A LAUNCH issues alone, so its device cycles are never added mid-group.
        if (windowDecoded[i].opcode == OPCODE_SYSTEM) {
            windowAluResult[i] = run_system(windowDecoded[i], windowWords[i], windowSlotPC[i]);
            continue;
        }
        read_one(windowDecoded[i], windowAluResult[i], windowMemRead[i], windowMemWrite[i], windowMemResult[i]);
    }
}

void CPU::writeback_m() {
    // Committing 0..m-1 in order is what resolves WAW correctly.
    for (int i = 0; i < issueCount; i++) {
        if (windowCancelled[i]) continue;
        if (windowRegWrite[i] && windowDecoded[i].rd) {
            registers[windowDecoded[i].rd] = windowMemRead[i] ? windowMemResult[i] : windowAluResult[i];
        }
    }
    enforce_zero_register();
    pc = windowNextPC;
    cycle_count++;
}
