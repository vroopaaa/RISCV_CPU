// NPU MMIO matmul test -- run manually via `emul 32I npu_matmul_test.c`.
// Not wired into run_tests_list.txt: this only pokes MMIO registers and
// doesn't check a result value in x10. A read of PRINT_ADDR dumps A/B/C to
// stdout, so this walks through the NPU's state at each stage:
//   1. before anything is loaded (everything zero)
//   2. after A/B are uploaded but before TRIGGER (C still zero)
//   3. after TRIGGER (C now holds the result)
//   4. after RESET (back to all zero)
#include <stdint.h>

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

#define NPU_BASE      0x80000000U
#define MAX_DIM       16U

#define DIM_M_ADDR    (NPU_BASE + 0x00)
#define DIM_K_ADDR    (NPU_BASE + 0x04)
#define DIM_N_ADDR    (NPU_BASE + 0x08)
#define TRIGGER_ADDR  (NPU_BASE + 0x0C)
#define MAC_ADDR      (NPU_BASE + 0x10)
#define RESET_ADDR    (NPU_BASE + 0x14)
#define STATUS_ADDR   (NPU_BASE + 0x18)
#define PRINT_ADDR    (NPU_BASE + 0x1C)

#define MAT_A_ADDR    (NPU_BASE + 0x100)
#define MAT_B_ADDR    (MAT_A_ADDR + MAX_DIM * MAX_DIM * 4)

int main() {
    // 3x2 * 2x4 -> 3x4: deliberately non-square and smaller than MAX_DIM,
    // to exercise the MAX_DIM row-stride addressing.
    const uint32_t M = 3, K = 2, N = 4;

    volatile int32_t A[3 * 2] = {
        1, 2,
        3, 4,
        5, 6,
    };
    volatile int32_t B[2 * 4] = {
        1, 0, 2, 1,
        0, 1, 1, 2,
    };


    npu_write32(DIM_M_ADDR, M);
    npu_write32(DIM_K_ADDR, K);
    npu_write32(DIM_N_ADDR, N);

    for (uint32_t i = 0; i < M; i++){
        for (uint32_t k = 0; k < K; k++){
            npu_write32(MAT_A_ADDR + (i * MAX_DIM + k) * 4, (uint32_t)A[i * K + k]);
        }
    }

    for (uint32_t k = 0; k < K; k++)
        for (uint32_t j = 0; j < N; j++)
            npu_write32(MAT_B_ADDR + (k * MAX_DIM + j) * 4, (uint32_t)B[k * N + j]);


    npu_write32(MAC_ADDR, 1);
    npu_read32(PRINT_ADDR);
    npu_write32(MAC_ADDR, 1);

    npu_read32(PRINT_ADDR); // 3: triggered -- C should now hold the matmul result

    return 0;
}
