// NPU custom-opcode bulk-transfer test -- run manually via
// `emul 32I npu_matmul_test_256.c`. Not wired into run_tests_list.txt: it's
// exercising a custom instruction rather than producing a single x10 value
// through the normal ISA, though it does self-check and return a mismatch
// count in a0 (needs a large cycle budget -- the O(16^3) mismatch-check
// loop alone runs past 100k cycles).
//
// Exercises the custom-0 (opcode 0x0B) NPU tile-transfer family added in
// CPU.cpp, which replaces a 256-word npu_write32 loop with a single
// instruction:
//   .insn r 0x0B, 0, 0, x0, rs1, rs2  -- load the 16x16 tile stored
//                                        contiguously at mem[rs1] into
//                                        Matrix A (rs2 unused)
//   .insn r 0x0B, 1, 0, x0, rs1, rs2  -- same, into Matrix B
//   .insn r 0x0B, 2, 0, x0, rs1, rs2  -- store Matrix C into mem[rs1], rows
//                                        rs2 bytes apart (strided write)
//   .insn r 0x0B, 3, 0, x0, rs1, rs2  -- print the NxN region at mem[rs1]
//                                        (rs2 = N), via Memory::print_matrix
// A/B/C here are each one full 16x16 tile (contiguous by construction), so
// this test can't tell a contiguous load apart from a strided one -- see
// matmul_template_2.c for a case that actually exercises tiling.
#include <stdint.h>

static inline void npu_write32(uint32_t addr, uint32_t val) {
    asm volatile ("sw %0, 0(%1)" : : "r"(val), "r"(addr) : "memory");
}

static inline void npu_load_a(const int32_t* src, uint32_t row_stride_bytes) {
    asm volatile (".insn r 0x0B, 0, 0, zero, %0, %1" : : "r"(src), "r"(row_stride_bytes) : "memory");
}

static inline void npu_load_b(const int32_t* src, uint32_t row_stride_bytes) {
    asm volatile (".insn r 0x0B, 1, 0, zero, %0, %1" : : "r"(src), "r"(row_stride_bytes) : "memory");
}

static inline void npu_store_c(int32_t* dst, uint32_t row_stride_bytes) {
    asm volatile (".insn r 0x0B, 2, 0, zero, %0, %1" : : "r"(dst), "r"(row_stride_bytes) : "memory");
}

// Prints the n x n region of memory at src (e.g. the result matrix, once
// npu_store_c has placed it there) -- a general memory print, not NPU state.
static inline void npu_print(const int32_t* src, uint32_t n) {
    asm volatile (".insn r 0x0B, 3, 0, zero, %0, %1" : : "r"(src), "r"(n) : "memory");
}

#define NPU_BASE      0x80000000U
#define MAX_DIM       16U

#define DIM_M_ADDR    (NPU_BASE + 0x00)
#define DIM_K_ADDR    (NPU_BASE + 0x04)
#define DIM_N_ADDR    (NPU_BASE + 0x08)
#define TRIGGER_ADDR  (NPU_BASE + 0x0C)
#define MAC_ADDR      (NPU_BASE + 0x10)

static int32_t A[MAX_DIM * MAX_DIM];
static int32_t B[MAX_DIM * MAX_DIM];
static int32_t C[MAX_DIM * MAX_DIM];

int main() {
    for (uint32_t i = 0; i < MAX_DIM * MAX_DIM; i++) {
        A[i] = (int32_t)(i % 11) - 5;
        B[i] = (int32_t)(i % 7) - 3;
    }

    npu_write32(DIM_M_ADDR, MAX_DIM);
    npu_write32(DIM_K_ADDR, MAX_DIM);
    npu_write32(DIM_N_ADDR, MAX_DIM);

    const uint32_t stride = MAX_DIM * sizeof(int32_t); // used only by npu_store_c; loads ignore it
    npu_load_a(A, stride);
    npu_load_b(B, stride);

    npu_write32(MAC_ADDR, 1);

    npu_store_c(C, stride);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < MAX_DIM; i++) {
        for (uint32_t j = 0; j < MAX_DIM; j++) {
            int32_t expected = 0;
            for (uint32_t k = 0; k < MAX_DIM; k++)
                expected += A[i * MAX_DIM + k] * B[k * MAX_DIM + j];
            if (C[i * MAX_DIM + j] != expected) mismatches++;
        }
    }
    npu_print(C, MAX_DIM);
    return (int)mismatches; // 0 = pass
}
