// vit.c -- Sparse-ViT-Det's ViT encoder on CRUDA's legos (NPU + GPU).
// vit_baseline.c computes the same thing on the host CPU alone; run_vit.py
// runs both and compares their outputs and cycles.
//
// The model so far, on VIT_ROWS x VIT_DIM activations (VIT_NUM_BATCHES
// batches stacked):
//   1. XN  = LayerNorm(X), no learned weights    GPU: one thread per row,
//                                                32-row blocks spread over the SMs
//   2. QKV = XN . Wqkv, no bias                   NPU: 16 x 16 tiles
//   3. S   = Q_h . K_h^T / 8 for every batch b    NPU: nn_gemm_bt, one call per
//      and head h; slot b * 6 + h of S            (b, h), on views into QKV
//   4. P   = softmax of every row of S            GPU: one call on all the
//                                                stacked rows, one thread per row
//   5. O   = P_h . V_h for every batch b and      NPU: nn_gemm, one call per
//      head h, the heads side by side in O        (b, h), on views into P, QKV, O
//   6. A   = O . Wo, no bias                      NPU: nn_gemm
//   7. X1  = X + A (first residual)               GPU: nn_add, one thread per element
//   8. XN2 = LayerNorm(X1), no learned weights    GPU: as step 1
//   9. H   = XN2 . W1, no bias (MLP first layer)  NPU: nn_gemm
//  10. G   = GELU(H)                              GPU: nn_lut8, one thread per element
//  11. F   = G . W2, no bias (MLP second layer)   NPU: nn_gemm
//  12. Y   = X1 + F (second residual)             GPU: nn_add, one thread per element
//
// Every tensor is in CRUDA's padded 16 x 16 tile layout (nn.h). Compiled
// -O2; the GPU lego files -O0.
#include <stdint.h>
#include "nn.h"
#include "vit.h"

_Static_assert(VIT_ROWS % NN_TILE == 0 && VIT_DIM % NN_TILE == 0,
               "sizes are whole tiles, so the tensors have no padding");

// Tiles per tile-row: ldt of a VIT_DIM-wide, a VIT_QKV-wide and a
// VIT_BATCH-wide (score) tensor.
#define VIT_LDT     (VIT_DIM / NN_TILE)
#define VIT_LDT_QKV (VIT_QKV / NN_TILE)
#define VIT_LDT_S   (VIT_BATCH / NN_TILE)
#define VIT_LDT_HID (VIT_HIDDEN / NN_TILE)

// A view of a tiled tensor: its tile (tr, tc), where the tensor has ldt
// tiles per tile-row. A sub-matrix starting on a tile boundary is just this
// pointer plus the parent's ldt -- nothing is copied.
#define VIT_TILE(base, ldt, tr, tc) ((base) + ((tr) * (ldt) + (tc)) * NN_TILE_BYTES)

// The input and the weights, already in tile order (make_input.py).
static const int8_t X[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include VIT_X_TILE_MAJOR
};
_Static_assert(sizeof X == VIT_ROWS * VIT_DIM, "the X file doesn't match vit.h's sizes");

static const int8_t Wqkv[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include "input/tile_major/Wqkv_384_1152.txt"
};
_Static_assert(sizeof Wqkv == VIT_DIM * VIT_QKV, "the Wqkv file doesn't match vit.h's sizes");

static const int8_t Wo[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include "input/tile_major/Wo_384_384.txt"
};
_Static_assert(sizeof Wo == VIT_DIM * VIT_DIM, "the Wo file doesn't match vit.h's sizes");

static const int8_t W1[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include "input/tile_major/W1_384_1536.txt"
};
_Static_assert(sizeof W1 == VIT_DIM * VIT_HIDDEN, "the W1 file doesn't match vit.h's sizes");

static const int8_t W2[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include "input/tile_major/W2_1536_384.txt"
};
_Static_assert(sizeof W2 == VIT_HIDDEN * VIT_DIM, "the W2 file doesn't match vit.h's sizes");

// nn_layernorm takes per-column gamma and beta; no weights means gamma = 1.0
// and beta = 0 in every column.
static int8_t  ln_gamma[VIT_DIM];
static int32_t ln_beta[VIT_DIM];

static int8_t XN[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));    // LayerNorm output
static int8_t QKV[VIT_ROWS * VIT_QKV] __attribute__((aligned(NN_TILE_BYTES)));   // Q | K | V
static int8_t S[VIT_S_ROWS * VIT_BATCH] __attribute__((aligned(NN_TILE_BYTES))); // scores
static int8_t P[VIT_S_ROWS * VIT_BATCH] __attribute__((aligned(NN_TILE_BYTES))); // probabilities
static int8_t O[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));     // head outputs
static int8_t A[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));     // output projection
static int8_t X1[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));    // first residual
static int8_t XN2[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));   // LN2's output

static int8_t H[VIT_ROWS * VIT_HIDDEN] __attribute__((aligned(NN_TILE_BYTES)));  // FC1's output
static int8_t G[VIT_ROWS * VIT_HIDDEN] __attribute__((aligned(NN_TILE_BYTES)));  // GELU's output
static int8_t F[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));     // FC2's output

static const uint16_t exp_table[] = {
#include "input/exp_table_256.txt"
};
_Static_assert(sizeof exp_table == 256 * sizeof(uint16_t), "the exp table must have 256 values");

static const int8_t gelu_table[] = {
#include "input/gelu_table_256.txt"
};
_Static_assert(sizeof gelu_table == 256, "the GELU table must have 256 values");

// Read by run_vit.py after the run: the output (tile layout), its size, and
// the cycles of the whole model and of each step.
int8_t         vit_out[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));   // the block's output Y
const uint32_t vit_out_rows = VIT_ROWS;
const uint32_t vit_out_cols = VIT_DIM;
uint64_t       vit_cycles;
uint64_t       vit_cycles_layernorm1;
uint64_t       vit_cycles_qkv;
uint64_t       vit_cycles_scores;
uint64_t       vit_cycles_softmax;
uint64_t       vit_cycles_pv;
uint64_t       vit_cycles_proj;
uint64_t       vit_cycles_residual1;
uint64_t       vit_cycles_layernorm2;
uint64_t       vit_cycles_fc1;
uint64_t       vit_cycles_gelu;
uint64_t       vit_cycles_fc2;
uint64_t       vit_cycles_residual2;

// Step 3. For batch b and head h, in tiles (tile = 16 x 16):
//   Q_h  = QKV rows of batch b, columns h * 64 ..       2 x 4 tiles at tile (2b, 4h)
//   K_h  = the same rows, columns 384 + h * 64 ..      2 x 4 tiles at tile (2b, 24 + 4h)
//   S_bh = Q_h . K_h^T                                  2 x 2 tiles, slot b * 6 + h of S
// nn_gemm_bt reads K_h as the stored N x K matrix of K_h^T (4 x 2 tiles):
// it walks along K_h's tile-rows and transposes each tile as the NPU loads
// it (load_b_t), so K^T is never written out.
static int attention_scores(const int8_t* QKV, int8_t* S, uint32_t shift) {
    const uint32_t bt = VIT_BATCH / NN_TILE;       // tile-rows per batch: 2
    const uint32_t ht = VIT_HEAD_DIM / NN_TILE;    // tile-columns per head: 4
    const uint32_t kt = VIT_DIM / NN_TILE;         // tile-column where K starts: 24
    int r = 0;
    for (uint32_t b = 0; b < VIT_NUM_BATCHES; b++) {
        for (uint32_t h = 0; h < VIT_HEADS; h++) {
            const int8_t* q = VIT_TILE(QKV, VIT_LDT_QKV, b * bt, h * ht);
            const int8_t* k = VIT_TILE(QKV, VIT_LDT_QKV, b * bt, kt + h * ht);
            int8_t*       s = VIT_TILE(S, VIT_LDT_S, (b * VIT_HEADS + h) * bt, 0);
            r |= nn_gemm_bt(q, VIT_LDT_QKV, k, VIT_LDT_QKV, 0, s, VIT_LDT_S,
                            VIT_BATCH, VIT_HEAD_DIM, VIT_BATCH, shift);
        }
    }
    return r;
}

// Step 5. For batch b and head h, in tiles:
//   P_bh = slot b * 6 + h of P                          2 x 2 tiles at tile (2 * slot, 0)
//   V_h  = QKV rows of batch b, columns 768 + h * 64 .. 2 x 4 tiles at tile (2b, 48 + 4h)
//   O_bh = P_bh . V_h                                    2 x 4 tiles at tile (2b, 4h) of O
// V_h is already K x N (keys x head columns), so this is plain nn_gemm, and
// writing each head into its own columns of O concatenates the heads.
static int weighted_values(const int8_t* P, const int8_t* QKV, int8_t* O, uint32_t shift) {
    const uint32_t bt = VIT_BATCH / NN_TILE;       // tile-rows per batch: 2
    const uint32_t ht = VIT_HEAD_DIM / NN_TILE;    // tile-columns per head: 4
    const uint32_t vt = 2 * VIT_DIM / NN_TILE;     // tile-column where V starts: 48
    int r = 0;
    for (uint32_t b = 0; b < VIT_NUM_BATCHES; b++) {
        for (uint32_t h = 0; h < VIT_HEADS; h++) {
            const int8_t* p = VIT_TILE(P, VIT_LDT_S, (b * VIT_HEADS + h) * bt, 0);
            const int8_t* v = VIT_TILE(QKV, VIT_LDT_QKV, b * bt, vt + h * ht);
            int8_t*       o = VIT_TILE(O, VIT_LDT, b * bt, h * ht);
            r |= nn_gemm(p, VIT_LDT_S, v, VIT_LDT_QKV, 0, o, VIT_LDT,
                         VIT_BATCH, VIT_BATCH, VIT_HEAD_DIM, shift);
        }
    }
    return r;
}

int main(void) {
    for (uint32_t c = 0; c < VIT_DIM; c++) {
        ln_gamma[c] = VIT_LN_ONE;
        ln_beta[c]  = 0;
    }

    uint64_t t0 = nn_cycles();
    int r = nn_layernorm(X, VIT_LDT, XN, VIT_LDT, VIT_ROWS, VIT_DIM, ln_gamma, ln_beta, 0);
    uint64_t t1 = nn_cycles();
    r |= nn_gemm(XN, VIT_LDT, Wqkv, VIT_LDT_QKV, 0,
                 QKV, VIT_LDT_QKV, VIT_ROWS, VIT_DIM, VIT_QKV, VIT_QKV_SHIFT);
    uint64_t t2 = nn_cycles();
    r |= attention_scores(QKV, S, VIT_SCORE_SHIFT);
    uint64_t t3 = nn_cycles();
    r |= nn_softmax(S, VIT_LDT_S, P, VIT_LDT_S, VIT_S_ROWS, VIT_BATCH, exp_table);
    uint64_t t4 = nn_cycles();
    r |= weighted_values(P, QKV, O, VIT_PV_SHIFT);
    uint64_t t5 = nn_cycles();
    r |= nn_gemm(O, VIT_LDT, Wo, VIT_LDT, 0,
                 A, VIT_LDT, VIT_ROWS, VIT_DIM, VIT_DIM, VIT_PROJ_SHIFT);
    uint64_t t6 = nn_cycles();
    r |= nn_add(X, VIT_LDT, 0, A, VIT_LDT, 0, X1, VIT_LDT, VIT_ROWS, VIT_DIM, VIT_RES_SHIFT);
    uint64_t t7 = nn_cycles();
    r |= nn_layernorm(X1, VIT_LDT, XN2, VIT_LDT, VIT_ROWS, VIT_DIM, ln_gamma, ln_beta, 0);
    uint64_t t8 = nn_cycles();
    r |= nn_gemm(XN2, VIT_LDT, W1, VIT_LDT_HID, 0,
                 H, VIT_LDT_HID, VIT_ROWS, VIT_DIM, VIT_HIDDEN, VIT_FC1_SHIFT);
    uint64_t t9 = nn_cycles();
    r |= nn_lut8(H, VIT_LDT_HID, G, VIT_LDT_HID, VIT_ROWS, VIT_HIDDEN, gelu_table);
    uint64_t t10 = nn_cycles();
    r |= nn_gemm(G, VIT_LDT_HID, W2, VIT_LDT, 0,
                 F, VIT_LDT, VIT_ROWS, VIT_HIDDEN, VIT_DIM, VIT_FC2_SHIFT);
    uint64_t t11 = nn_cycles();
    r |= nn_add(X1, VIT_LDT, VIT_RES2_X1_UP, F, VIT_LDT, 0,
                vit_out, VIT_LDT, VIT_ROWS, VIT_DIM, VIT_RES2_SHIFT);
    uint64_t t12 = nn_cycles();

    vit_cycles_layernorm1 = t1 - t0;
    vit_cycles_qkv       = t2 - t1;
    vit_cycles_scores    = t3 - t2;
    vit_cycles_softmax   = t4 - t3;
    vit_cycles_pv        = t5 - t4;
    vit_cycles_proj      = t6 - t5;
    vit_cycles_residual1 = t7 - t6;
    vit_cycles_layernorm2 = t8 - t7;
    vit_cycles_fc1       = t9 - t8;
    vit_cycles_gelu      = t10 - t9;
    vit_cycles_fc2       = t11 - t10;
    vit_cycles_residual2 = t12 - t11;
    vit_cycles           = t12 - t0;
    return r;
}
