// Same NxN * NxN NPU matmul as matmul_template_2.c (same tile-major A/B
// layout, same custom-0 helpers, same register map) -- the only difference
// is the inner K-tile loop's body, which is filled in by
// run_npu_tests_2.py's Python-side unroller instead of being a real C `for`
// loop here. See that script for why: each K-tile iteration currently ends
// in a `bne` back to the loop top, and hazard_scan()'s control-hazard rule
// (see docs/superscalar.md) closes the superscalar issue window at every
// single one of those, regardless of issue width -- unlike a RAW or memory
// hazard, a wider window can never route around a branch. Unrolling removes
// the branch itself for whichever K-tiles get flattened, instead of trying
// to schedule around it.
//
// The BIG_DIM/MATRIX_A_DATA/MATRIX_B_DATA/KT_LOOP_BODY placeholder tokens
// below (each wrapped in double curly braces) are filled in by
// run_npu_tests_2.py -- this file is never compiled as-is.
#include <stdint.h>

// ---------------------------------------------------------------------
// Low-level MMIO + custom-instruction helpers (identical to
// matmul_template_2.c)
// ---------------------------------------------------------------------

static inline void npu_write32(uint32_t addr, uint32_t val) {
    asm volatile (
        "sw %0, 0(%1)"
        :
        : "r"(val), "r"(addr)
        : "memory"
    );
}

static inline void npu_load_a(const int32_t* src) {
    asm volatile (".insn r 0x0B, 0, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

static inline void npu_load_b(const int32_t* src) {
    asm volatile (".insn r 0x0B, 1, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

static inline void npu_store_c(int32_t* dst, uint32_t row_stride_bytes) {
    asm volatile (".insn r 0x0B, 2, 0, zero, %0, %1" : : "r"(dst), "r"(row_stride_bytes) : "memory");
}

static inline void npu_print(const int32_t* src, uint32_t n) {
    asm volatile (".insn r 0x0B, 3, 0, zero, %0, %1" : : "r"(src), "r"(n) : "memory");
}

// ---------------------------------------------------------------------
// NPU register map (matches NPU::write()/read() in NPU.cpp)
// ---------------------------------------------------------------------

#define NPU_BASE      0x80000000U
#ifndef MAX_DIM
#define MAX_DIM       {{TILE_DIM}}U
#endif

#define DIM_M_ADDR    (NPU_BASE + 0x00)
#define DIM_K_ADDR    (NPU_BASE + 0x04)
#define DIM_N_ADDR    (NPU_BASE + 0x08)
#define TRIGGER_ADDR  (NPU_BASE + 0x0C)
#define MAC_ADDR      (NPU_BASE + 0x10)
#define RESET_ADDR    (NPU_BASE + 0x14)
#define STATUS_ADDR   (NPU_BASE + 0x18)

#define BIG_DIM       {{BIG_DIM}}U
#define TILES         (BIG_DIM / MAX_DIM)
#define TILE_WORDS    (MAX_DIM * MAX_DIM)
#define ROW_STRIDE    (BIG_DIM * sizeof(int32_t))

#define RESULT_BASE_ADDR 0xC8000U

static const int32_t A[BIG_DIM * BIG_DIM] = {
{{MATRIX_A_DATA}}
};

static const int32_t B[BIG_DIM * BIG_DIM] = {
{{MATRIX_B_DATA}}
};

int main() {
    const uint32_t M = MAX_DIM, K = MAX_DIM, N = MAX_DIM;

    // Every tile here is 16x16 (M=K=N=MAX_DIM), so the dims never actually
    // change across the whole run -- set them once. reset() (see NPU.cpp)
    // deliberately leaves M/K/N alone, only clearing A/B/C and `done`, so
    // this doesn't need to be re-declared inside the tile loop the way it
    // used to.
    npu_write32(DIM_M_ADDR, M);
    npu_write32(DIM_K_ADDR, K);
    npu_write32(DIM_N_ADDR, N);

    for (uint32_t I = 0; I < TILES; I++) {
        for (uint32_t J = 0; J < TILES; J++) {

            npu_write32(RESET_ADDR, 1);

            // K-tile loop -- unrolled by run_npu_tests_2.py, see file header.
{{KT_LOOP_BODY}}

            npu_store_c((int32_t*)(RESULT_BASE_ADDR + ((I * MAX_DIM) * BIG_DIM + J * MAX_DIM) * 4), ROW_STRIDE);
        }
    }

    npu_print((int32_t*)RESULT_BASE_ADDR, BIG_DIM);

    return 0;
}
