#include "../include/CPU.h"
#include "../include/NPU.h"
#include <iostream>
#include <cstring>
#include <iomanip>

// Constructor: Initializes the CPU state
CPU::CPU(Memory* mem_ptr) {
    memory = mem_ptr;

    std::memset(registers, 0, sizeof(registers));
    pc = 0;
    cycle_count = 0;
    halted = false;
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
        // Memory::print_matrix(). All done in read(), touching neither a
        // register nor the scalar mem_read/write path.
        case 0x0B:
            mem_read_enable  = false;
            mem_write_enable = false;
            reg_write_enable = false;
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
    // Custom-0 (opcode 0x0B): NPU tile transfer (funct3 0/1/2) or a general
    // memory print (funct3 3). rs1 = base address.
    //
    // funct3 0/1 (loads) read the source tile as one contiguous 256-word
    // block -- the caller is responsible for laying tiles out contiguously
    // in memory ahead of time, so this is a single access rather than 16
    // separate strided row reads. rs2 is unused here.
    //
    // funct3 2 (store) still writes strided, since the destination is a
    // plain row-major matrix (rows are BIG_DIM apart, not tile-contiguous):
    // rs2 = row stride in bytes.
    //
    // funct3 3 prints an NxN region (rs2 = N) via Memory::print_matrix,
    // unrelated to the NPU -- just a general "print memory" instruction.
    // ---------------------------------------------------------
    if (decodedInstruction.opcode == 0x0B) {
        uint32_t base = aluResult;
        uint32_t rs2_val = registers[decodedInstruction.rs2];
        switch (decodedInstruction.funct3) {
            case 0x0: // load Matrix A: memory -> NPU (tile is contiguous in memory)
                for (uint32_t i = 0; i < NPU::MAX_DIM * NPU::MAX_DIM; i++)
                    memory->write_word(NPU::MAT_A_ADDR + i * 4, memory->read_word(base + i * 4));
                break;
            case 0x1: // load Matrix B: memory -> NPU (tile is contiguous in memory)
                for (uint32_t i = 0; i < NPU::MAX_DIM * NPU::MAX_DIM; i++)
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
        }
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
