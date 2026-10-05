#include "../include/NPU.h"
#include <iostream>

NPU::NPU() : M(0), K(0), N(0), done(false) {
    for (uint32_t i = 0; i < 2 * TILE_ELEMS; i++) ab[i] = 0;
    for (uint32_t i = 0; i < TILE_ELEMS; i++)     c[i]  = 0;
}

void NPU::compute() {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += (int32_t)ab[A_OFFSET + i * MAX_DIM + k] * (int32_t)ab[B_OFFSET + k * MAX_DIM + j];
            c[i * MAX_DIM + j] = sum;
        }
    done = true;    
}

void NPU::mac() {
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < K; k++)
                sum += (int32_t)ab[A_OFFSET + i * MAX_DIM + k] * (int32_t)ab[B_OFFSET + k * MAX_DIM + j];
            c[i * MAX_DIM + j] += sum;
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

    if (address >= MAT_A_ADDR && address < RESULT_ADDR) {
        // A/B are int8, so one word write fills 4 consecutive elements
        // (little-endian). Only A/B are CPU-writable; the result is output-only.
        uint32_t byte_off = (address - MAT_A_ADDR) & ~3u;
        for (uint32_t i = 0; i < 4; i++)
            ab[byte_off + i] = (int8_t)((val >> (8 * i)) & 0xFF);
        return;
    }
    if (address >= RESULT_ADDR && address < DATA_WINDOW_END) {
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

    if (address >= MAT_A_ADDR && address < RESULT_ADDR) {
        uint32_t byte_off = (address - MAT_A_ADDR) & ~3u;   // pack 4 int8 -> 1 word
        uint32_t word = 0;
        for (uint32_t i = 0; i < 4; i++)
            word |= (uint32_t)(uint8_t)ab[byte_off + i] << (8 * i);
        return word;
    }
    if (address >= RESULT_ADDR && address < DATA_WINDOW_END) {
        return (uint32_t)c[(address - RESULT_ADDR) / 4];
    }

    std::cerr << "[NPU] Read from unmapped or write-only register at 0x"
              << std::hex << address << std::dec << "\n";
    return 0;
}

void NPU::reset() {
    // Clears the matrices
    done = false;
    for (uint32_t i = 0; i < 2 * TILE_ELEMS; i++) ab[i] = 0;
    for (uint32_t i = 0; i < TILE_ELEMS; i++)     c[i]  = 0;
}