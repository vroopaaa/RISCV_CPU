// vit.h -- sizes, number formats and data files shared by vit.c (CRUDA) and
// vit_baseline.c (plain C on the host CPU). The data files are in
// input/row_major/ and input/tile_major/ (make_input.py).
#ifndef VIT_H
#define VIT_H

// The activation matrix is VIT_NUM_BATCHES batches of VIT_BATCH rows,
// stacked: VIT_ROWS x VIT_DIM. Change VIT_NUM_BATCHES to pick another input.
#define VIT_NUM_BATCHES 8u
#define VIT_BATCH       32u     // rows per batch
#define VIT_ROWS        (VIT_NUM_BATCHES * VIT_BATCH)
#define VIT_DIM         384u    // embedding width
#define VIT_QKV         (3u * VIT_DIM)   // Q | K | V side by side

// The input file for that batch count, X_<rows>_384.txt, in each layout.
#if VIT_NUM_BATCHES == 1
#define VIT_X_ROW_MAJOR  "input/row_major/X_32_384.txt"
#define VIT_X_TILE_MAJOR "input/tile_major/X_32_384.txt"
#elif VIT_NUM_BATCHES == 2
#define VIT_X_ROW_MAJOR  "input/row_major/X_64_384.txt"
#define VIT_X_TILE_MAJOR "input/tile_major/X_64_384.txt"
#elif VIT_NUM_BATCHES == 4
#define VIT_X_ROW_MAJOR  "input/row_major/X_128_384.txt"
#define VIT_X_TILE_MAJOR "input/tile_major/X_128_384.txt"
#elif VIT_NUM_BATCHES == 6
#define VIT_X_ROW_MAJOR  "input/row_major/X_192_384.txt"
#define VIT_X_TILE_MAJOR "input/tile_major/X_192_384.txt"
#elif VIT_NUM_BATCHES == 8
#define VIT_X_ROW_MAJOR  "input/row_major/X_256_384.txt"
#define VIT_X_TILE_MAJOR "input/tile_major/X_256_384.txt"
#else
#error "no input file for this VIT_NUM_BATCHES (make_input.py writes 1, 2, 4, 6 and 8)"
#endif

// Number formats: an int8 value q stands for q * 2^-f, f = fractional bits
// (vit_block.md section 5).
//
// LayerNorm output: the normalized value (zero mean, unit variance) at
// f = VIT_LN_FRAC, so int8 covers -4 .. +3.97 in steps of 1/32.
// No learned weights: gamma is 1.0 (VIT_LN_ONE at this scale), beta 0.
#define VIT_LN_FRAC  5u
#define VIT_LN_ONE   (1 << VIT_LN_FRAC)

// QKV projection: QKV = LN(X) . Wqkv, no bias. Wqkv has f = 9, so the int32
// sum has f = 5 + 9 = 14; shifting right by 9 leaves Q, K, V at f = 5.
#define VIT_QKV_SHIFT 9u

#endif // VIT_H
