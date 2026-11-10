// vit.c -- Sparse-ViT-Det's ViT encoder on CRUDA's legos (NPU + GPU).
// vit_baseline.c computes the same thing on the host CPU alone; run_vit.py
// runs both and compares their outputs and cycles.
//
// Step 1: LayerNorm of VIT_NUM_BATCHES batches (VIT_ROWS x VIT_DIM, stacked)
// with no learned weights, in one call: each 32-row block of rows goes to an SM.
//
// Every tensor is in CRUDA's padded 16 x 16 tile layout (nn.h). Compiled
// -O2; the GPU lego files -O0.
#include <stdint.h>
#include "nn.h"
#include "vit.h"

_Static_assert(VIT_ROWS % NN_TILE == 0 && VIT_DIM % NN_TILE == 0,
               "sizes are whole tiles, so the tensors have no padding");

#define VIT_LDT (VIT_DIM / NN_TILE)    // tiles per tile-row of a VIT_DIM-wide tensor

// The input, already in tile order (make_input.py).
#define VIT_INPUT_TILED
static const int8_t X[] __attribute__((aligned(NN_TILE_BYTES))) = {
#include VIT_INPUT_FILE
};
_Static_assert(sizeof X == VIT_ROWS * VIT_DIM, "the input file doesn't match vit.h's sizes");

// nn_layernorm takes per-column gamma and beta; no weights means gamma = 1.0
// and beta = 0 in every column.
static int8_t  ln_gamma[VIT_DIM];
static int32_t ln_beta[VIT_DIM];

// Read by run_vit.py after the run: the output (tile layout), its size, and
// the cycles spent in the model.
int8_t         vit_out[VIT_ROWS * VIT_DIM] __attribute__((aligned(NN_TILE_BYTES)));
const uint32_t vit_out_rows = VIT_ROWS;
const uint32_t vit_out_cols = VIT_DIM;
uint64_t       vit_cycles;

int main(void) {
    for (uint32_t c = 0; c < VIT_DIM; c++) {
        ln_gamma[c] = VIT_LN_ONE;
        ln_beta[c]  = 0;
    }

    uint64_t t0 = nn_cycles();
    int r = nn_layernorm(X, VIT_LDT, vit_out, VIT_LDT, VIT_ROWS, VIT_DIM, ln_gamma, ln_beta, 0);
    uint64_t t1 = nn_cycles();

    vit_cycles = t1 - t0;
    return r;
}
