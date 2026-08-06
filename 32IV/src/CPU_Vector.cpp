#include <cstdint>
#include <algorithm>
#include "CPU.h"

void CPU::vector_decode() {
    vectorInstruction.funct6 = (instruction >> 26) & 0x3F; 
    vectorInstruction.funct3 = (instruction >> 12) & 0x07; 
    vectorInstruction.vd = (instruction >> 7) & 0x1F;      
    vectorInstruction.rd = (instruction >> 7) & 0x1F;      
    vectorInstruction.vs1 = (instruction >> 15) & 0x1F;    
    vectorInstruction.rs1 = (instruction >> 15) & 0x1F;    
    vectorInstruction.vs2 = (instruction >> 20) & 0x1F;    
    vectorInstruction.rs2 = (instruction >> 20) & 0x1F;    
    vectorInstruction.vm = (instruction >> 25) & 0x01;     
    vectorInstruction.imm_Z_VSETVLI = (instruction >> 20) & 0x7FF; 
    vectorInstruction.imm_Z_VSETIVLI = (instruction >> 20) & 0x3FF; 
    vectorInstruction.imm_U = (instruction >> 15) & 0x1F;  
    vectorInstruction.bit_31 = (instruction >> 31) & 0x01; 
    vectorInstruction.bit_30 = (instruction >> 30) & 0x01; 
    reg_write_enable = true; 
    mem_read_enable = false; 
    mem_write_enable = false; 
}

void CPU::vector_config_execute() {
    enum VectorConfigType {
        VSETVLI,
        VSETIVLI,
        VSETVL 
    };
    
    VectorConfigType configType;
    uint32_t proposed_vtype;

    if (!vectorInstruction.bit_31) {
        configType = VSETVLI;
        proposed_vtype = vectorInstruction.imm_Z_VSETVLI;
    } 
    else if (vectorInstruction.bit_30) { 
        configType = VSETIVLI;
        proposed_vtype = vectorInstruction.imm_Z_VSETIVLI;
    } 
    else {
        configType = VSETVL;
        proposed_vtype = registers[vectorInstruction.rs2]; 
    }

    uint8_t vtype_sew = (proposed_vtype >> 3) & 0x7;
    uint8_t vtype_lmul = proposed_vtype & 0x7;
    
    if (vtype_sew > 0x3 || vtype_lmul == 0x4) {
        vtype = 0x80000000; 
        vl = 0;
        aluResult = 0; 
        return; 
    }
    
    float lmul;
    uint32_t sew = 8 * (1 << vtype_sew);
    
    switch(vtype_lmul) {
        case 0x5: lmul = 0.125f; break;
        case 0x6: lmul = 0.25f;  break;
        case 0x7: lmul = 0.5f;   break;  
        default:  lmul = static_cast<float>(1 << vtype_lmul); break;
    }
    
    uint32_t vlmax = static_cast<uint32_t>((VLEN * lmul) / sew);

    uint32_t avl;
    if (configType == VSETIVLI) {
        avl = vectorInstruction.imm_U; 
    } 
    else {
        if (vectorInstruction.rs1 == 0x0) {
            avl = vlmax; 
        } else {
            avl = registers[vectorInstruction.rs1];
        }
    }

    uint32_t new_vl;
    if (avl <= vlmax) {
        new_vl = avl;
    } else {
        new_vl = vlmax;
    }

    vtype = proposed_vtype; 
    vl = new_vl; 
    aluResult = new_vl; 
}

void CPU::vector_read() {
    // Memory/Data Read phase placeholder
}

void CPU::vector_writeback() {
    if (reg_write_enable && vectorInstruction.rd != 0x0) {
        if (mem_read_enable) {
            registers[vectorInstruction.rd] = memResult; 
        } else {
            registers[vectorInstruction.rd] = aluResult; 
        }
    }
}