// vit.h -- sizes and number formats shared by vit.c (CRUDA) and
// vit_baseline.c (plain C on the host CPU).
#ifndef VIT_H
#define VIT_H

// The activation matrix is VIT_NUM_BATCHES batches of VIT_BATCH rows,
// stacked: VIT_ROWS x VIT_DIM. Change VIT_NUM_BATCHES to pick another input.
#define VIT_NUM_BATCHES 4u
#define VIT_BATCH       32u     // rows per batch
#define VIT_ROWS        (VIT_NUM_BATCHES * VIT_BATCH)
#define VIT_DIM         384u    // embedding width

// The input file for that batch count (make_input.py). Each file has both
// orders: tiled when VIT_INPUT_TILED is defined, row-major otherwise.
#if VIT_NUM_BATCHES == 1
#define VIT_INPUT_FILE "vit_input_b1.txt"
#elif VIT_NUM_BATCHES == 2
#define VIT_INPUT_FILE "vit_input_b2.txt"
#elif VIT_NUM_BATCHES == 4
#define VIT_INPUT_FILE "vit_input_b4.txt"
#elif VIT_NUM_BATCHES == 6
#define VIT_INPUT_FILE "vit_input_b6.txt"
#elif VIT_NUM_BATCHES == 8
#define VIT_INPUT_FILE "vit_input_b8.txt"
#else
#error "no input file for this VIT_NUM_BATCHES (make_input.py writes 1, 2, 4, 6 and 8)"
#endif

// LayerNorm output: the normalized value (zero mean, unit variance) at
// VIT_LN_FRAC fractional bits, so int8 covers -4 .. +3.97 in steps of 1/32.
// No learned weights: gamma is 1.0 (VIT_LN_ONE at this scale), beta 0.
#define VIT_LN_FRAC  5u
#define VIT_LN_ONE   (1 << VIT_LN_FRAC)

#endif // VIT_H
