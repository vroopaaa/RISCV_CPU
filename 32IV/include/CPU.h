#ifndef CPU_H
#define CPU_H

#include <cstdint>
#include "memory.h"

// Define the architecture parameters
#define XLEN 32
#define NUM_REGS 32
typedef uint32_t reg_t;



class CPU {
private:
    reg_t registers[NUM_REGS]; // The 32 general-purpose registers
    reg_t pc;                  // The Program Counter
    reg_t next_pc;             // The next Program Counter (for branch handling)
    reg_t instruction;          // The current instruction being executed
    reg_t aluResult;            // Result from the ALU operation
    reg_t memResult;            // Result from memory read operation
    Memory* memory;            // Pointer to the memory object
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
    struct VectorInstructionFields {
        uint8_t funct6;
        uint8_t funct3;
        uint8_t vd;
        uint8_t rd;
        uint8_t vs1;
        uint8_t rs1;
        uint8_t vs2;
        uint8_t rs2;
        bool vm;
        uint32_t imm_Z_VSETVLI;
        uint32_t imm_Z_VSETIVLI;
        uint32_t imm_U;
        bool bit_31;
        bool bit_30;
    };
    VectorInstructionFields vectorInstruction;
    reg_t vstart, vxsat, vxrm, vcsr, vl, vtype, vlenb; 
public:
    // Constructor
    CPU(Memory* mem_ptr);

    // Core execution pipeline methods
    void fetch();
    void decode();
    void execute();
    void read();
    void writeback();
    void vector_decode();
    void vector_config_execute();
    void vector_read();
    void vector_writeback();
    // Debugging method to print the CPU state
    void print_state();
};

#endif // CPU_H