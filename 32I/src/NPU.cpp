#include "../include/NPU.h"
#include <iostream>

NPU::NPU() : M(0), K(0), N(0), done(false) {
    for (uint32_t i = 0; i < 3 * MAX_DIM * MAX_DIM; i++)
        data[i] = 0;
}

void NPU::compute() {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += data[A_OFFSET + i * MAX_DIM + k] * data[B_OFFSET + k * MAX_DIM + j];
            data[C_OFFSET + i * MAX_DIM + j] = sum;
        }
    done = true;    
}

void NPU::mac() {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += data[A_OFFSET + i * MAX_DIM + k] * data[B_OFFSET + k * MAX_DIM + j];
            data[C_OFFSET + i * MAX_DIM + j] += sum;
        }
    done = true;
}

void NPU::write(uint32_t address, uint32_t val) {
    if (address == DIM_M_ADDR)   { M = val > MAX_DIM ? MAX_DIM : val; return; }
    if (address == DIM_K_ADDR)   { K = val > MAX_DIM ? MAX_DIM : val; return; }
    if (address == DIM_N_ADDR)   { N = val > MAX_DIM ? MAX_DIM : val; return; }
    if (address == TRIGGER_ADDR) { compute(); return; }
    if (address == MAC_ADDR)     { mac(); return; }
    if (address == RESET_ADDR)   { reset(); return; }

    if (address >= MAT_A_ADDR && address < DATA_WINDOW_END) {
        uint32_t idx = (address - MAT_A_ADDR) / 4;   // one index space covers A, B, C
        if (idx < C_OFFSET) {   // only A/B are CPU-writable; result is output-only
            data[idx] = (int32_t)val;
            return;
        }
        std::cerr << "[NPU] Write to read-only RESULT register at 0x"
                  << std::hex << address << std::dec << " ignored\n";
        return;
    }

    std::cerr << "[NPU] Write to unmapped register at 0x"
              << std::hex << address << std::dec << " ignored\n";
}

uint32_t NPU::read(uint32_t address) {
    if (address == STATUS_ADDR) return done ? 1u : 0u;
    if (address == PRINT_ADDR)  { print_matrices(); return 0; }

    if (address >= MAT_A_ADDR && address < DATA_WINDOW_END) {
        uint32_t idx = (address - MAT_A_ADDR) / 4;
        return (uint32_t)data[idx];
    }

    std::cerr << "[NPU] Read from unmapped or write-only register at 0x"
              << std::hex << address << std::dec << "\n";
    return 0;
}

void NPU::reset() {
    // Clears the matrices
    done = false;
    for (uint32_t i = 0; i < 3 * MAX_DIM * MAX_DIM; i++)
        data[i] = 0;
}