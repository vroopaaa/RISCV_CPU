// Template for a square NxN * NxN NPU matmul test, tiled into 16x16 blocks.
// The BIG_DIM/MATRIX_A_DATA/MATRIX_B_DATA placeholder tokens below (each
// wrapped in double curly braces) are filled in by run_npu_tests.py -- this
// file is never compiled as-is.
#include <stdint.h>

// ---------------------------------------------------------------------
// Low-level MMIO helpers
// ---------------------------------------------------------------------

static inline void npu_write32(uint32_t addr, uint32_t val) {
    asm volatile (
        "sw %0, 0(%1)"
        :
        : "r"(val), "r"(addr)
        : "memory"
    );
}

static inline uint32_t npu_read32(uint32_t addr) {
    uint32_t val;
    asm volatile (
        "lw %0, 0(%1)"
        : "=r"(val)
        : "r"(addr)
        : "memory"
    );
    return val;
}

// Move one word directly from an NPU MMIO register into ordinary Memory.
// Early-clobber ("=&r") on tmp: it's written by the lw before it's read by
// the sw, so tmp must not share a register with either input address.
static inline void npu_move_word(uint32_t npu_addr, uint32_t mem_addr) {
    uint32_t tmp;
    asm volatile (
        "lw %0, 0(%1)\n\t"
        "sw %0, 0(%2)"
        : "=&r"(tmp)
        : "r"(npu_addr), "r"(mem_addr)
        : "memory"
    );
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
#define RESET_ADDR    (NPU_BASE + 0x14)   // clears A/B/C and done; leaves M/K/N alone
#define STATUS_ADDR   (NPU_BASE + 0x18)
#define PRINT_ADDR    (NPU_BASE + 0x1C)

#define MAT_A_ADDR    (NPU_BASE + 0x100)
#define MAT_B_ADDR    (MAT_A_ADDR + MAX_DIM * MAX_DIM * 4)
#define RESULT_ADDR   (MAT_B_ADDR + MAX_DIM * MAX_DIM * 4)

// ---------------------------------------------------------------------
// Problem size: BIG_DIM x BIG_DIM * BIG_DIM x BIG_DIM -> BIG_DIM x BIG_DIM,
// tiled as a (BIG_DIM/16) grid of 16x16 tiles. BIG_DIM must be a multiple
// of MAX_DIM (16) -- run_npu_tests.py is responsible for only ever
// generating multiples of 16 here.
// ---------------------------------------------------------------------

#define BIG_DIM       {{BIG_DIM}}U
#define TILES         (BIG_DIM / MAX_DIM)

// Fixed address (~800KB into the 1M RAM) instead of a stack-managed one, so
// the result can be located and read back after the run via
// Memory::dump_range without reading a map file.
#define RESULT_BASE_ADDR 0xC8000U

// static const: lives directly in the binary's data image, no runtime copy
// needed (unlike a local array's initializer, which C requires to be
// re-copied onto the stack on every call -- wasteful here since main()
// only runs once). The reads below still can't be optimized away, since
// they feed into npu_write32's "memory"-clobbering inline asm.
static const int32_t A[BIG_DIM * BIG_DIM] = {
{{MATRIX_A_DATA}}
};

static const int32_t B[BIG_DIM * BIG_DIM] = {
{{MATRIX_B_DATA}}
};

int main() {
    const uint32_t M = MAX_DIM, K = MAX_DIM, N = MAX_DIM; // every tile is 16x16 here

    // Every tile here is 16x16 (M=K=N=MAX_DIM), so the dims never actually
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
                // Load A[I][Kt] tile (strided out of the BIG_DIM-wide big matrix)
                for (uint32_t i = 0; i < M; i++) {
                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t val = (uint32_t)A[(I * MAX_DIM + i) * BIG_DIM + (Kt * MAX_DIM + k)];
                        npu_write32(MAT_A_ADDR + (i * MAX_DIM + k) * 4, val);
                    }
                }

                // Load B[Kt][J] tile
                for (uint32_t k = 0; k < K; k++) {
                    for (uint32_t j = 0; j < N; j++) {
                        uint32_t val = (uint32_t)B[(Kt * MAX_DIM + k) * BIG_DIM + (J * MAX_DIM + j)];
                        npu_write32(MAT_B_ADDR + (k * MAX_DIM + j) * 4, val);
                    }
                }

                // Accumulate this K-slice's contribution on-chip.
                npu_write32(MAC_ADDR, 1);
            }

            // K-loop finished: drain the completed 16x16 accumulator straight
            // into the result matrix's (I,J) tile, at its fixed base address
            // in ordinary Memory -- no pointer needed, same as MAT_A_ADDR etc.
            for (uint32_t i = 0; i < M; i++) {
                for (uint32_t j = 0; j < N; j++) {
                    npu_move_word(
                        RESULT_ADDR + (i * MAX_DIM + j) * 4,
                        RESULT_BASE_ADDR + ((I * MAX_DIM + i) * BIG_DIM + (J * MAX_DIM + j)) * 4
                    );
                }
            }
        }
    }

    npu_read32(PRINT_ADDR);

    return 0;
}
