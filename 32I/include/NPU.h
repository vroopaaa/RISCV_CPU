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
    static constexpr uint32_t MAX_DIM         = 16;

    static constexpr uint32_t DIM_M_ADDR      = NPU_BASE + 0x00;
    static constexpr uint32_t DIM_K_ADDR      = NPU_BASE + 0x04;
    static constexpr uint32_t DIM_N_ADDR      = NPU_BASE + 0x08;
    static constexpr uint32_t TRIGGER_ADDR    = NPU_BASE + 0x0C;
    static constexpr uint32_t MAC_ADDR        = NPU_BASE + 0x10; // triggers a MAC instead of a compute
    static constexpr uint32_t RESET_ADDR      = NPU_BASE + 0x14;
    // bit0: DONE -- set once compute() finishes, cleared by a write to RESET_ADDR
    static constexpr uint32_t STATUS_ADDR     = NPU_BASE + 0x18;
    // Read-triggered: any read here dumps A/B/C to stdout; the returned value is unused
    static constexpr uint32_t PRINT_ADDR      = NPU_BASE + 0x1C;


    // A, B, and result all live inside ONE contiguous data window now
    static constexpr uint32_t MAT_A_ADDR      = NPU_BASE + 0x100;
    static constexpr uint32_t MAT_B_ADDR      = MAT_A_ADDR + MAX_DIM * MAX_DIM * 4;  
    static constexpr uint32_t RESULT_ADDR     = MAT_B_ADDR + MAX_DIM * MAX_DIM * 4;  
    static constexpr uint32_t DATA_WINDOW_END = RESULT_ADDR + MAX_DIM * MAX_DIM * 4; 
    static constexpr uint32_t NPU_WINDOW_SIZE = DATA_WINDOW_END - NPU_BASE;

private:
    // one flat array = the NPU's entire data SRAM.
    // A occupies [0, 64), B occupies [64, 128), C (result) occupies [128, 192).
    int32_t data[3 * MAX_DIM * MAX_DIM];

    static constexpr uint32_t A_OFFSET = 0;
    static constexpr uint32_t B_OFFSET = MAX_DIM * MAX_DIM;
    static constexpr uint32_t C_OFFSET = 2 * MAX_DIM * MAX_DIM;

    uint32_t M, K, N;
    bool done;
    void compute();
    void mac();
    void reset();
};
#endif