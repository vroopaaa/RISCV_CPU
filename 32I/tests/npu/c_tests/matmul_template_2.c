// Template for a square NxN * NxN NPU matmul test, tiled into 16x16 blocks,
// using the custom-0 (opcode 0x0B) tile-transfer instructions in place of
// matmul_template.c's per-word npu_write32/npu_move_word loops.
//
// A and B are stored TILE-MAJOR (each 16x16 tile contiguous, tiles in
// row-major grid order), not as plain row-major BIG_DIM x BIG_DIM arrays --
// that lets npu_load_a/npu_load_b read a whole tile in one contiguous
// access instead of 16 separate strided row reads. The result matrix is
// still written out plain row-major (via a strided store) since nothing
// downstream needs it tiled.
//
// The BIG_DIM/MATRIX_A_DATA/MATRIX_B_DATA placeholder tokens below (each
// wrapped in double curly braces) are filled in by run_npu_tests.py, which
// is responsible for pre-arranging A/B into this tile-major order -- this
// file is never compiled as-is.
#include <stdint.h>

// ---------------------------------------------------------------------
// Low-level MMIO + custom-instruction helpers
// ---------------------------------------------------------------------

static inline void npu_write32(uint32_t addr, uint32_t val) {
    asm volatile (
        "sw %0, 0(%1)"
        :
        : "r"(val), "r"(addr)
        : "memory"
    );
}

// Loads a 16x16 int8 tile stored CONTIGUOUSLY in memory (256 bytes starting
// at src) into the NPU's Matrix A/B window -- a single contiguous access,
// unlike npu_store_c below which still has to land into a strided,
// row-major destination. Each call replaces what used to be a per-word
// npu_write32/npu_move_word loop per tile.
static inline void npu_load_a(const int8_t* src) {
    asm volatile (".insn r 0x0B, 0, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

static inline void npu_load_b(const int8_t* src) {
    asm volatile (".insn r 0x0B, 1, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

// Stores the NPU's 16x16 result tile into memory at dst, rows
// row_stride_bytes apart -- the destination result matrix stays plain
// row-major, so this side keeps the strided write.

static inline void npu_store_c(int32_t* dst, uint32_t row_stride_bytes) {
    asm volatile (".insn r 0x0B, 2, 0, zero, %0, %1" : : "r"(dst), "r"(row_stride_bytes) : "memory");
}

// Prints the n x n region of memory at src (e.g. the fully-assembled
// result matrix at RESULT_BASE_ADDR) -- a general memory print, not NPU
// internal state, so it shows the whole BIG_DIM x BIG_DIM result rather
// than just whichever 16x16 tile the NPU last touched.
static inline void npu_print(const int32_t* src, uint32_t n) {
    asm volatile (".insn r 0x0B, 3, 0, zero, %0, %1" : : "r"(src), "r"(n) : "memory");
}

// ---------------------------------------------------------------------
// NPU register map (matches NPU::write()/read() in NPU.cpp)
// ---------------------------------------------------------------------

#define NPU_BASE      0x80000000U
#ifndef MAX_DIM
#define MAX_DIM       {{TILE_DIM}}U          // NPU's native tile size
#endif

#define DIM_M_ADDR    (NPU_BASE + 0x00)
#define DIM_K_ADDR    (NPU_BASE + 0x04)
#define DIM_N_ADDR    (NPU_BASE + 0x08)
#define TRIGGER_ADDR  (NPU_BASE + 0x0C)   // one-shot overwrite compute
#define MAC_ADDR      (NPU_BASE + 0x10)   // accumulate compute
#define RESET_ADDR    (NPU_BASE + 0x14)   // clears A/B/C and done; leaves M/K/N alone
#define STATUS_ADDR   (NPU_BASE + 0x18)

// ---------------------------------------------------------------------
// Problem size: BIG_DIM x BIG_DIM * BIG_DIM x BIG_DIM -> BIG_DIM x BIG_DIM,
// tiled as a (BIG_DIM/MAX_DIM) grid of MAX_DIMxMAX_DIM tiles. BIG_DIM must be
// a multiple of MAX_DIM.
// ---------------------------------------------------------------------

#define BIG_DIM       {{BIG_DIM}}U
#define TILES         (BIG_DIM / MAX_DIM)
#define TILE_ELEMS    (MAX_DIM * MAX_DIM)
// Only the result store still uses a row stride -- it's the one thing left
// plain row-major (see file header comment). C is int32, so 4 bytes/element.
#define ROW_STRIDE    (BIG_DIM * sizeof(int32_t))

// Fixed address (~800KB into the 1M RAM) instead of a stack-managed one, so
// the result can be located and read back after the run via
// Memory::dump_range without reading a map file.
#define RESULT_BASE_ADDR 0xC8000U

// Tile-major: tile (row, col) of the logical BIG_DIM x BIG_DIM matrix lives
// at [(row * TILES + col) * TILE_ELEMS, ...), 256 contiguous int8 elements,
// row-major within the tile. A/B are int8; the result C is int32. Filled in
// by run_npu_tests.py in this order.
static const int8_t A[BIG_DIM * BIG_DIM] __attribute__((aligned(4))) = {
{{MATRIX_A_DATA}}
};

static const int8_t B[BIG_DIM * BIG_DIM] __attribute__((aligned(4))) = {
{{MATRIX_B_DATA}}
};

int main() {
    const uint32_t M = MAX_DIM, K = MAX_DIM, N = MAX_DIM; // every tile is MAX_DIMxMAX_DIM here

    // Every tile here is MAX_DIMxMAX_DIM (M=K=N=MAX_DIM), so the dims never actually
    // change across the whole run -- set them once. reset() (see NPU.cpp)
    // deliberately leaves M/K/N alone, only clearing A/B/C and `done`, so
    // this doesn't need to be re-declared inside the tile loop.
    npu_write32(DIM_M_ADDR, M);
    npu_write32(DIM_K_ADDR, K);
    npu_write32(DIM_N_ADDR, N);

    for (uint32_t I = 0; I < TILES; I++) {
        for (uint32_t J = 0; J < TILES; J++) {

            // Start each output tile with a clean accumulator.
            npu_write32(RESET_ADDR, 1);

            for (uint32_t Kt = 0; Kt < TILES; Kt++) {
                // Load A[I][Kt] and B[Kt][J] tiles straight out of the
                // tile-major big matrices -- each tile is TILE_ELEMS contiguous
                // int8 elements, so this is one single access, not strided rows.
                npu_load_a(&A[(I * TILES + Kt) * TILE_ELEMS]);
                npu_load_b(&B[(Kt * TILES + J) * TILE_ELEMS]);

                // Accumulate this K-slice's contribution on-chip.
                npu_write32(MAC_ADDR, 1);
            }

            // K-loop finished: drain the completed accumulator
            // straight into the result matrix's (I,J) tile -- one
            // instruction, same strided addressing as the loads above.
            npu_store_c((int32_t*)(RESULT_BASE_ADDR + ((I * MAX_DIM) * BIG_DIM + J * MAX_DIM) * 4), ROW_STRIDE);
        }
    }

    npu_print((int32_t*)RESULT_BASE_ADDR, BIG_DIM);

    return 0;
}
