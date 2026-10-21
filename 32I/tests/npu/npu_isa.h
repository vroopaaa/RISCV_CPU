// npu_isa.h -- C helpers for the NPU: its MMIO registers and the custom-0
// (opcode 0x0B) tile-transfer instructions, so programs don't redefine them.
// Same role as tests/simt/simt_isa.h for the SIMT GPU. Addresses must match
// include/NPU.h.
//
// The NPU computes C (16x16 int32) = A (16x16 int8) x B (16x16 int8) over
// the DIM_M x DIM_N region, summing DIM_K terms (TRIGGER overwrites C, MAC
// adds to it). Tiles are 256 contiguous bytes, row-major inside the tile.
// A tile starts from RESET (C = 0) or from load_bias (C(i, j) = bias[j] in
// every row), and the output stage turns C into int8:
//   out(i, j) = sat8(rshift(C(i, j), shift))  if i < M and j < N
//             = 0                              otherwise
// with rshift rounding halves toward +inf.
#ifndef NPU_ISA_H
#define NPU_ISA_H

#include <stdint.h>

#define NPU_BASE        0x80000000u
#define NPU_DIM_M       (NPU_BASE + 0x00)
#define NPU_DIM_K       (NPU_BASE + 0x04)
#define NPU_DIM_N       (NPU_BASE + 0x08)
#define NPU_TRIGGER     (NPU_BASE + 0x0C)   // C = A x B
#define NPU_MAC         (NPU_BASE + 0x10)   // C += A x B
#define NPU_RESET       (NPU_BASE + 0x14)   // clears A/B/C/done; keeps dims, shift
#define NPU_STATUS      (NPU_BASE + 0x18)
#define NPU_PRINT       (NPU_BASE + 0x1C)   // a read dumps A/B/C to stdout
#define NPU_OUT_SHIFT   (NPU_BASE + 0x20)   // output-stage shift (low 5 bits)

#define NPU_TILE_DIM    16u
#define NPU_TILE_ELEMS  (NPU_TILE_DIM * NPU_TILE_DIM)
#define NPU_MAT_A       (NPU_BASE + 0x100)
#define NPU_MAT_B       (NPU_MAT_A + NPU_TILE_ELEMS)          // int8
#define NPU_RESULT      (NPU_MAT_B + NPU_TILE_ELEMS)          // int32 C
#define NPU_BIAS        (NPU_RESULT + NPU_TILE_ELEMS * 4)     // 16 x int32, write-only: word j -> C column j
#define NPU_OUT8        (NPU_BIAS + NPU_TILE_DIM * 4)         // int8 output stage, read-only

static inline void npu_write32(uint32_t addr, uint32_t val) {
    *(volatile uint32_t*)addr = val;
}

static inline uint32_t npu_read32(uint32_t addr) {
    return *(volatile uint32_t*)addr;
}

// funct3 0: copy a 256-byte int8 tile from src into A.
static inline void npu_load_a(const int8_t* src) {
    asm volatile(".insn r 0x0B, 0, 0, zero, %0, zero" : : "r"(src) : "memory");
}

// funct3 1: copy a 256-byte int8 tile from src into B.
static inline void npu_load_b(const int8_t* src) {
    asm volatile(".insn r 0x0B, 1, 0, zero, %0, zero" : : "r"(src) : "memory");
}

// funct3 2: store the int32 C tile at dst, rows row_stride_bytes apart.
static inline void npu_store_c(int32_t* dst, uint32_t row_stride_bytes) {
    asm volatile(".insn r 0x0B, 2, 0, zero, %0, %1" : : "r"(dst), "r"(row_stride_bytes) : "memory");
}

// funct3 3: debug print of an n x n int32 region of memory at src.
static inline void npu_print(const void* src, uint32_t n) {
    asm volatile(".insn r 0x0B, 3, 0, zero, %0, %1" : : "r"(src), "r"(n) : "memory");
}

// funct3 4: preload C from 16 int32 bias values at src (bias[j] down every row).
static inline void npu_load_bias(const int32_t* src) {
    asm volatile(".insn r 0x0B, 4, 0, zero, %0, zero" : : "r"(src) : "memory");
}

// funct3 5: run the output stage on C and store the 256-byte int8 tile at dst.
static inline void npu_store_c8(int8_t* dst, uint32_t shift) {
    asm volatile(".insn r 0x0B, 5, 0, zero, %0, %1" : : "r"(dst), "r"(shift) : "memory");
}

#endif // NPU_ISA_H
