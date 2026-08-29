#include "../include/NPU.h"
#include <iostream>
#include <iomanip>

// data/M/K/N/A_OFFSET/etc. are NPU-private; this lives in its own file for
// separation from the register read/write logic in NPU.cpp, but it's still a
// member function so it can reach them directly.
static void print_one(const char* name, const int32_t* base, uint32_t rows, uint32_t cols) {
    std::cout << "[NPU] " << name << " (" << rows << "x" << cols << "):\n";
    for (uint32_t i = 0; i < rows; i++) {
        std::cout << "  ";
        for (uint32_t j = 0; j < cols; j++)
            std::cout << std::setw(6) << base[i * NPU::MAX_DIM + j];
        std::cout << "\n";
    }
}

void NPU::print_matrices() const {
    // CPU::print_state() leaves cout in hex/zero-fill mode; reset it so these
    // matrices print as plain decimal columns.
    std::cout << std::dec << std::setfill(' ');
    print_one("A", &data[A_OFFSET], M, K);
    print_one("B", &data[B_OFFSET], K, N);
    print_one("C", &data[C_OFFSET], M, N);
}
