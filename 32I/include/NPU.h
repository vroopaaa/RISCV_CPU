#ifndef NPU_H
#define NPU_H
#include <cstdint>

class NPU {
public:
    NPU();
    void write(uint32_t address, uint32_t data);
    uint32_t read(uint32_t address);
    // Dumps A/B/C (defined in NPU_print.cpp); triggered by a read of PRINT_ADDR.
    void print_matrices() const;

    static constexpr uint32_t NPU_BASE        = 0x80000000;
#ifndef NPU_MAX_DIM
#define NPU_MAX_DIM 16
#endif
    static constexpr uint32_t MAX_DIM         = NPU_MAX_DIM;

    static constexpr uint32_t DIM_M_ADDR      = NPU_BASE + 0x00;
    static constexpr uint32_t DIM_K_ADDR      = NPU_BASE + 0x04;
    static constexpr uint32_t DIM_N_ADDR      = NPU_BASE + 0x08;
    static constexpr uint32_t TRIGGER_ADDR    = NPU_BASE + 0x0C;
    static constexpr uint32_t MAC_ADDR        = NPU_BASE + 0x10; // triggers a MAC instead of a compute
    static constexpr uint32_t RESET_ADDR      = NPU_BASE + 0x14; // clears A/B/C and done; leaves M/K/N alone (see reset())
    // bit0: DONE -- set once compute() finishes, cleared by a write to RESET_ADDR
    static constexpr uint32_t STATUS_ADDR     = NPU_BASE + 0x18;
    // Read-triggered: any read here dumps A/B/C to stdout; the returned value is unused
    static constexpr uint32_t PRINT_ADDR      = NPU_BASE + 0x1C;


    // A, B, and result all live inside ONE contiguous data window.
    // A and B are int8 (1 byte/element, packed 4 per word over MMIO,
    // little-endian); the result C is int32 (4 bytes/element) since
    // accumulating K int8 products overflows 8 bits.
    static constexpr uint32_t TILE_ELEMS      = MAX_DIM * MAX_DIM;
    static_assert(TILE_ELEMS % 4 == 0, "A/B tiles must be a whole number of words (MAX_DIM even)");
    static constexpr uint32_t MAT_A_ADDR      = NPU_BASE + 0x100;
    static constexpr uint32_t MAT_B_ADDR      = MAT_A_ADDR + TILE_ELEMS;      // int8
    static constexpr uint32_t RESULT_ADDR     = MAT_B_ADDR + TILE_ELEMS;      // int32 from here
    static constexpr uint32_t DATA_WINDOW_END = RESULT_ADDR + TILE_ELEMS * 4;
    static constexpr uint32_t NPU_WINDOW_SIZE = DATA_WINDOW_END - NPU_BASE;

private:
    // The NPU's data SRAM: int8 operands (A then B, contiguous) and the
    // int32 accumulator/result C.
    int8_t  ab[2 * TILE_ELEMS];   // A occupies [0, TILE_ELEMS), B occupies [TILE_ELEMS, 2*TILE_ELEMS)
    int32_t c[TILE_ELEMS];

    static constexpr uint32_t A_OFFSET = 0;
    static constexpr uint32_t B_OFFSET = TILE_ELEMS;

    uint32_t M, K, N;
    bool done;
    void compute();
    void mac();
    void reset();
};
#endif