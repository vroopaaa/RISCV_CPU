#include "../include/CPU.h"
#include <iostream>
#include <cstring>
#include <iomanip>

// Constructor: Initializes the CPU state
CPU::CPU(Memory* mem_ptr) {
    memory = mem_ptr;

    std::memset(registers, 0, sizeof(registers));
    pc = 0;
}

// Enforces that register x0 is always hardwired to 0
void CPU::enforce_zero_register() {
    registers[0] = 0;
}

// A handy debug function to see what is inside your registers
void CPU::print_state() {
    std::cout << "--- CPU State ---" << std::endl;
    std::cout << "PC: 0x" << std::hex << std::setfill('0') << std::setw(8) << pc << std::endl;
    
    for (int i = 0; i < NUM_REGS; i++) {
        std::cout << "x" << std::dec << i << ": 0x" 
                  << std::hex << std::setfill('0') << std::setw(8) << registers[i] << "  ";
        if ((i + 1) % 4 == 0) std::cout << std::endl; // Print 4 registers per row
    }
    std::cout << "-----------------" << std::endl;
}

// Stubs for your pipeline methods (to be filled in next)
void CPU::fetch() {
    // Logic to read from memory using PC and store in instruction
    // update the PC to point to the next instruction
    // if it sees isBranchTaken, then it should update the PC to branchPC instead of incrementing by 4
    // also update the decode and execute methods
    instruction = memory->read_word(pc);
    pc += 4; // Move to the next instruction (assuming 4-byte instructions)
}

void CPU::decode() {
    // Logic to extract opcode, rd, rs1, rs2, etc.
    // Get the opcode
    uint8_t opcode = instruction & 0x7F; // Extract the last 7 bits for opcode
    switch (opcode) {
        //Integer Register-Immediate Instructions (I-type)
        // 0010011 ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI
        case 0x13: {
            // Extract fields for I-type instruction
            uint8_t rd = (instruction >> 7) & 0x1F; // bits 11-7    
            uint8_t funct3 = (instruction >> 12) & 0x07; // bits 14-12
            uint8_t rs1 = (instruction >> 15) & 0x1F; // bits 19-15
            int32_t imm = (int32_t)(instruction) >> 20; // bits 31-20, sign-extended
            //ADDI adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored and
            //the result is simply the low XLEN bits of the result. ADDI rd, rs1, 0 is used to implement the MV
            //rd, rs1 assembler pseudo-instruction.
            switch (funct3) {
                case 0x0: // ADDI
                    registers[rd] = registers[rs1] + imm;
                    break;
                case 0x2: // SLTI
                    registers[rd] = (registers[rs1] < imm) ? 1 : 0;
                    break;
                case 0x3: // SLTIU
                    registers[rd] = ((uint32_t)registers[rs1] < (uint32_t)imm) ? 1 : 0;
                    break;
                case 0x4: // XORI
                    registers[rd] = registers[rs1] ^ imm;
                    break;
                case 0x6: // ORI
                    registers[rd] = registers[rs1] | imm;
                    break;
                case 0x7: // ANDI
                    registers[rd] = registers[rs1] & imm;
                    break;
                case 0x1: // SLLI
                    registers[rd] = registers[rs1] << (imm & 0x1F);
                    break;
                case 0x5: // SRLI and SRAI
                    if ((imm >> 10) & 0x1) { // Check if it's SRAI
                        registers[rd] = (int32_t)registers[rs1] >> (imm & 0x1F); // Arithmetic right shift
                    } else { // SRLI
                        registers[rd] = registers[rs1] >> (imm & 0x1F); // Logical right shift
                    }
                    break;
            }
        }
        

    }
}

void CPU::execute() {
    // Switch statement for instruction execution
    
    // Always enforce x0=0 at the end of execution
    enforce_zero_register();
}

void CPU::read() {
    // Logic to read from registers or memory
}

void CPU::writeback() {
    // Logic to write results back to registers
}