// nn.h -- the lego library: shared definitions and every lego's host wrapper.
// Rules (number format, tile layout, conventions) are in docs/design/00-overview.md;
// each lego has its own design file next to it.
//
// Every tensor is a matrix stored as padded 16 x 16 tiles (256 contiguous
// bytes each, row-major inside the tile, tiles in row-major order) and is
// passed as a pointer to its first tile plus ldt, the number of tiles in one
// tile-row of the underlying buffer. Padding is always zero.
#ifndef NN_H
#define NN_H

#include <stdint.h>

#define NN_TILE         16u
#define NN_TILE_BYTES   (NN_TILE * NN_TILE)
#define NN_PAD16(x)     (((x) + NN_TILE - 1u) & ~(NN_TILE - 1u))

// GEMM on the NPU (docs/design/gemm.md):
//   Y = sat8(rshift(X . W + bias, shift))
// X is M x K, W is K x N, Y is M x N; bias is NN_PAD16(N) int32 or NULL.
// Returns 0, or -1 without touching the NPU if a parameter is invalid.
int nn_gemm(const int8_t*  X, uint32_t ldt_x,
            const int8_t*  W, uint32_t ldt_w,
            const int32_t* bias,
            int8_t*        Y, uint32_t ldt_y,
            uint32_t M, uint32_t K, uint32_t N,
            uint32_t shift);

#endif // NN_H
