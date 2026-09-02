// Template for a square NxN * NxN NPU matmul test, tiled into 16x16 blocks,
// using the custom-0 (opcode 0x0B) tile-transfer instructions in place of
// matmul_template.c's per-word npu_write32/npu_move_word loops.
//
// A, B, AND the result are all stored TILE-MAJOR (each 16x16 tile
// contiguous, tiles in row-major grid order), not as plain row-major
// BIG_DIM x BIG_DIM arrays -- that lets npu_load_a/npu_load_b/npu_store_c
// each move a whole tile in one contiguous access instead of 16 separate
// strided row accesses. This means the result region at RESULT_BASE_ADDR is
// NOT a plain row-major matrix -- run_npu_tests.py de-tiles it back into
// logical (row, col) order before comparing against its Python-computed
// reference.
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

// Loads a 16x16 tile stored CONTIGUOUSLY in memory (256 words starting at
// src) into the NPU's Matrix A/B window -- a single contiguous access.
// Each call replaces what used to be a 256-word npu_write32/npu_move_word
// loop per tile.
static inline void npu_load_a(const int32_t* src) {
    asm volatile (".insn r 0x0B, 0, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

static inline void npu_load_b(const int32_t* src) {
    asm volatile (".insn r 0x0B, 1, 0, zero, %0, %1" : : "r"(src), "r"(0) : "memory");
}

// Stores the NPU's 16x16 result tile as 256 contiguous words at dst --
// same single-access shape as the loads above, landing into the result
// buffer's tile-major layout (see file header comment).
static inline void npu_store_c(int32_t* dst) {
    asm volatile (".insn r 0x0B, 2, 0, zero, %0, %1" : : "r"(dst), "r"(0) : "memory");
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
#define MAX_DIM       16U          // NPU's native tile size (16x16)

#define DIM_M_ADDR    (NPU_BASE + 0x00)
#define DIM_K_ADDR    (NPU_BASE + 0x04)
#define DIM_N_ADDR    (NPU_BASE + 0x08)
#define TRIGGER_ADDR  (NPU_BASE + 0x0C)   // one-shot overwrite compute
#define MAC_ADDR      (NPU_BASE + 0x10)   // accumulate compute
#define RESET_ADDR    (NPU_BASE + 0x14)   // clears A/B/C AND dims AND done
#define STATUS_ADDR   (NPU_BASE + 0x18)

// ---------------------------------------------------------------------
// Problem size: BIG_DIM x BIG_DIM * BIG_DIM x BIG_DIM -> BIG_DIM x BIG_DIM,
// tiled as a (BIG_DIM/16) grid of 16x16 tiles. BIG_DIM must be a multiple
// of MAX_DIM (16) -- run_npu_tests.py is responsible for only ever
// generating multiples of 16 here.
// ---------------------------------------------------------------------

#define BIG_DIM       {{BIG_DIM}}U
#define TILES         (BIG_DIM / MAX_DIM)
#define TILE_WORDS    (MAX_DIM * MAX_DIM)

// Fixed address (~800KB into the 1M RAM) instead of a stack-managed one, so
// the result can be located and read back after the run via
// Memory::dump_range without reading a map file.
#define RESULT_BASE_ADDR 0xC8000U

// Tile-major: tile (row, col) of the logical BIG_DIM x BIG_DIM matrix lives
// at [(row * TILES + col) * TILE_WORDS, ...), 256 contiguous elements,
// row-major within the tile. Filled in by run_npu_tests.py in this order.
static const int32_t A[BIG_DIM * BIG_DIM] = {
{{MATRIX_A_DATA}}
};

static const int32_t B[BIG_DIM * BIG_DIM] = {
{{MATRIX_B_DATA}}
};

int main() {
    const uint32_t M = MAX_DIM, K = MAX_DIM, N = MAX_DIM; // every tile is 16x16 here

    for (uint32_t I = 0; I < TILES; I++) {
        for (uint32_t J = 0; J < TILES; J++) {

            // Start each output tile with a clean accumulator.
            // NOTE: RESET_ADDR clears M/K/N too, so they must be re-declared
            // every time this is called -- not just once before the loop.
            npu_write32(RESET_ADDR, 1);
            npu_write32(DIM_M_ADDR, M);
            npu_write32(DIM_K_ADDR, K);
            npu_write32(DIM_N_ADDR, N);

            for (uint32_t Kt = 0; Kt < TILES; Kt++) {
                // Load A[I][Kt] and B[Kt][J] tiles straight out of the
                // tile-major big matrices -- each tile is 256 contiguous
                // words, so this is one single access, not 16 strided rows.
                npu_load_a(&A[(I * TILES + Kt) * TILE_WORDS]);
                npu_load_b(&B[(Kt * TILES + J) * TILE_WORDS]);

                // Accumulate this K-slice's contribution on-chip.
                npu_write32(MAC_ADDR, 1);
            }

            // K-loop finished: drain the completed 16x16 accumulator into
            // the result buffer's (I,J) tile slot -- one contiguous access,
            // same shape as the loads above. The result buffer is
            // tile-major (see file header comment), not row-major.
            npu_store_c((int32_t*)(RESULT_BASE_ADDR + (I * TILES + J) * TILE_WORDS * 4));
        }
    }

    // NOTE: this prints the raw tile-major buffer as if it were a BIG_DIM x
    // BIG_DIM row-major matrix, so for BIG_DIM > MAX_DIM the on-screen
    // layout is scrambled -- it's just a quick "did anything come out"
    // sanity view. run_npu_tests.py de-tiles the same buffer properly (via
    // Memory::dump_range) for the actual correctness check.
    npu_print((int32_t*)RESULT_BASE_ADDR, BIG_DIM);

    return 0;
}
