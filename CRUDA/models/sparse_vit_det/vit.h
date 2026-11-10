// vit.h -- sizes, number formats and data files shared by vit.c (CRUDA) and
// vit_baseline.c (plain C on the host CPU). The data files are in
// input/row_major/ and input/tile_major/ (make_input.py).
#ifndef VIT_H
#define VIT_H

// The activation matrix is VIT_NUM_BATCHES batches of VIT_BATCH rows,
// stacked: VIT_ROWS x VIT_DIM. Change VIT_NUM_BATCHES to pick another input.
#define VIT_NUM_BATCHES 6u
#define VIT_BATCH       32u     // rows per batch
#define VIT_ROWS        (VIT_NUM_BATCHES * VIT_BATCH)
#define VIT_DIM         384u    // embedding width
#define VIT_QKV         (3u * VIT_DIM)   // Q | K | V side by side

// Attention: VIT_HEADS heads of VIT_HEAD_DIM columns each. Q is columns
// 0 .. 383 of QKV, K 384 .. 767, V 768 .. 1151; head h is columns
// h * 64 .. h * 64 + 63 of each. Tokens attend only within their own batch,
// so there is one VIT_BATCH x VIT_BATCH score matrix per (batch, head).
#define VIT_HEADS       6u
#define VIT_HEAD_DIM    (VIT_DIM / VIT_HEADS)            // 64
#define VIT_SLOTS       (VIT_NUM_BATCHES * VIT_HEADS)    // score matrices
#define VIT_S_ROWS      (VIT_SLOTS * VIT_BATCH)          // all of them stacked

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

// Scores: S = Q_h . K_h^T / sqrt(64), per batch and head. Q and K have f = 5,
// so the int32 sum has f = 10; / sqrt(64) = / 8 is 3 more bits of shift, and
// shifting by 10 + 3 - 4 = 9 leaves the scores at f = 4 (-8 .. +7.94).
// sqrt(VIT_HEAD_DIM) must be a power of two for this.
#define VIT_SCORE_SHIFT 9u
_Static_assert(VIT_HEAD_DIM == 64, "VIT_SCORE_SHIFT assumes sqrt(head dim) = 8");

// Softmax along each score row (32 values: one query against its batch's 32
// keys): P = e / sum(e), e = table[max - s] from input/exp_table_256.txt
// (2^15 * e^(-d/16), for the scores' f = 4). P has f = 7: steps of 1/128,
// largest 127/128 (softmax.md).
#define VIT_P_FRAC   7u

#endif // VIT_H
