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
    reg_t instruction;          // The current instruction being executed
    Memory* memory;            // Pointer to the memory object
    // Helper method to enforce hardware rules
    void enforce_zero_register();
    bool isBranchTaken();
    reg_t branchPC;

public:
    // Constructor
    CPU(Memory* mem_ptr);

    // Core execution pipeline methods
    void fetch();
    void decode();
    void execute();
    void read();
    void writeback();
    // Debugging method to print the CPU state
    void print_state();
};

#endif // CPU_H