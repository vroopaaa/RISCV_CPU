#pragma once
#include <cstdint>
#include <array>

#define VLEN 512
#define VLEN_BYTES 32 
struct VectorRegister {
    std::array<std::array<uint8_t, VLEN_BYTES>, 32> vregs; // 32 vector registers, each 512 bits (64 bytes)
};


uint8_t vtype_sew, vtype_lmul, vtype_vill;



