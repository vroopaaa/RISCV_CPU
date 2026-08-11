#ifndef CPU_H
#define CPU_H

#include <cstdint>
#include <array>
#include "memory.h"
#include "CPU_Vector.h"

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
    bool vector_mem_enable;      // set for any vector load/store opcode (0x07/0x27); read()
                                  // routes to vector_read() off this signal instead of the
                                  // scalar mem enables, generalizing to future vector
                                  // memory instruction variants.
    bool vector_instruction_active; // set for any vector opcode (0x57/0x07/0x27); writeback()
                                  // routes to vector_writeback() off this signal instead of
                                  // the scalar reg_write_enable path, so vector_writeback()
                                  // is the single place that commits config/load/store
                                  // results (scalar or vector register file) uniformly.
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
        // Load/store-only fields (opcodes 0x07/0x27). For these opcodes `funct3` above
        // doubles as the `width` field, `vs2` doubles as lumop/sumop for unit-stride, the
        // stride GPR index for strided, or the index vector register for indexed; `vd`
        // doubles as `vs3` (store data source register / first field's register).
        uint8_t nf;   // NFIELDS = nf+1 (segments); also encodes register count for whole-reg
        bool mew;
        uint8_t mop;  // 00=unit-stride, 01=indexed-unordered, 10=strided, 11=indexed-ordered
        uint32_t eew_bytes;  // decoded from `width`: data EEW for unit-stride/strided, index
                              // EEW for indexed (indexed's data width is SEW, from vtype). 0=reserved.
        bool ls_supported;         // gates address-gen/read/writeback into a safe no-op
        bool ls_whole_register;    // lumop/sumop == 0b01000 (unit-stride only)
        bool ls_fault_only_first;  // lumop == 0b10000 (unit-stride load only)
    };
    VectorInstructionFields vectorInstruction;
    reg_t vstart, vxsat, vxrm, vcsr, vl, vtype, vlenb;
    VectorRegister vregfile; // The 32 vector registers; v0 doubles as the mask register
    // Scratch state carrying the load/store address-gen (execute) -> memory access (read)
    // -> commit (writeback) handoff, mirroring how aluResult/memResult carry scalar state
    // between stages. Sized for the worst case: EEW=8 gives VLEN_BYTES lanes/segments, and
    // segments span up to 8 fields (NFIELDS max), each up to VLEN_BYTES.
    std::array<bool, VLEN_BYTES> vec_lane_active;
    std::array<uint32_t, VLEN_BYTES> vec_lane_addr; // per-lane/segment base address
    uint32_t vec_num_lanes;
    std::array<std::array<uint8_t, VLEN_BYTES>, 8> vec_load_buffer; // [field][byte]
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
    void vector_load_store_execute();
    void vector_read();
    void vector_writeback();
    // Debugging method to print the CPU state
    void print_state();
    void print_vector_reg(uint8_t idx);
};

#endif // CPU_H