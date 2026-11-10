// GEMM legos on the NPU -- docs/design/gemm.md and docs/design/gemm_bt.md.
//
//   nn_gemm:     Y = sat8(rshift(X . W   + bias, shift))     W  stored K x N
//   nn_gemm_bt:  Y = sat8(rshift(X . Wt^T + bias, shift))    Wt stored N x K
//
// Host functions: they walk the output tile by tile and drive the NPU with
// the custom-0 instructions. Each output tile starts from its bias
// (load_bias: C = bias in every row) or from RESET (C = 0), accumulates one
// MAC per K tile, and leaves through the output stage (store_c8: shift,
// clamp, zero outside the DIM_M x DIM_N region). DIM_K stays 16: the K
// padding of X and W is zero, so a full tile adds nothing extra.
#include "../include/nn.h"
#include "../../32I/tests/npu/npu_isa.h"

// Shared by both wrappers. transposed = 0: W is K x N, tile (k, J) is read
// with load_b, walking down W's tile-column J. transposed = 1: W holds Wt
// (N x K); tile (k, J) of Wt^T is tile (J, k) of Wt, read with load_b_t,
// walking along Wt's tile-row J. always_inline: each wrapper gets its own
// copy with `transposed` a constant, so the unused K loop and the test are
// compiled away and nn_gemm costs exactly what it did before nn_gemm_bt.
static inline __attribute__((always_inline)) int gemm_core(const int8_t*  X, uint32_t ldt_x,
                     const int8_t*  W, uint32_t ldt_w,
                     const int32_t* bias,
                     int8_t*        Y, uint32_t ldt_y,
                     uint32_t M, uint32_t K, uint32_t N,
                     uint32_t shift, uint32_t transposed) {
    if (M == 0 || K == 0 || N == 0) return -1;
    if (X == 0 || W == 0 || Y == 0) return -1;
    if (((uint32_t)X | (uint32_t)W | (uint32_t)Y) & 3u) return -1;
    uint32_t Mt = NN_PAD16(M) / NN_TILE;
    uint32_t Kt = NN_PAD16(K) / NN_TILE;
    uint32_t Nt = NN_PAD16(N) / NN_TILE;
    uint32_t w_row_tiles = transposed ? Kt : Nt;   // tiles in one row of the stored W
    if (ldt_x < Kt || ldt_w < w_row_tiles || ldt_y < Nt || shift > 31) return -1;

    npu_write32(NPU_DIM_K, NN_TILE);
    for (uint32_t J = 0; J < Nt; J++) {
        uint32_t n_left = N - J * NN_TILE;
        npu_write32(NPU_DIM_N, n_left < NN_TILE ? n_left : NN_TILE);
        for (uint32_t I = 0; I < Mt; I++) {
            uint32_t m_left = M - I * NN_TILE;
            npu_write32(NPU_DIM_M, m_left < NN_TILE ? m_left : NN_TILE);
            if (bias)
                npu_load_bias(bias + J * NN_TILE);   // C = bias
            else
                npu_write32(NPU_RESET, 1);           // C = 0

            const int8_t* x_tile = X + I * ldt_x * NN_TILE_BYTES;   // tile (I, 0)
            if (!transposed) {
                const int8_t* w_tile = W + J * NN_TILE_BYTES;       // tile (0, J) of W
                for (uint32_t k = 0; k < Kt; k++) {
                    npu_load_a(x_tile);
                    npu_load_b(w_tile);
                    npu_write32(NPU_MAC, 1);                         // C += A . B
                    x_tile += NN_TILE_BYTES;
                    w_tile += ldt_w * NN_TILE_BYTES;                 // one tile down W
                }
            } else {
                const int8_t* w_tile = W + J * ldt_w * NN_TILE_BYTES; // tile (J, 0) of Wt
                for (uint32_t k = 0; k < Kt; k++) {
                    npu_load_a(x_tile);
                    npu_load_b_t(w_tile);                            // B = tile (J, k) of Wt, transposed
                    npu_write32(NPU_MAC, 1);                         // C += A . B
                    x_tile += NN_TILE_BYTES;
                    w_tile += NN_TILE_BYTES;                         // one tile along Wt
                }
            }
            npu_store_c8(Y + (I * ldt_y + J) * NN_TILE_BYTES, shift);
        }
    }
    return 0;
}

int nn_gemm(const int8_t*  X, uint32_t ldt_x,
            const int8_t*  W, uint32_t ldt_w,
            const int32_t* bias,
            int8_t*        Y, uint32_t ldt_y,
            uint32_t M, uint32_t K, uint32_t N,
            uint32_t shift) {
    return gemm_core(X, ldt_x, W, ldt_w, bias, Y, ldt_y, M, K, N, shift, 0);
}

int nn_gemm_bt(const int8_t*  X,  uint32_t ldt_x,
               const int8_t*  Wt, uint32_t ldt_wt,
               const int32_t* bias,
               int8_t*        Y,  uint32_t ldt_y,
               uint32_t M, uint32_t K, uint32_t N,
               uint32_t shift) {
    return gemm_core(X, ldt_x, Wt, ldt_wt, bias, Y, ldt_y, M, K, N, shift, 1);
}
