// c_identity -- every thread records its identity values; 9 blocks on 4 SMs,
// 40 threads per block (second warp partly masked).
#include <stdint.h>
#include "simt_isa.h"

#define THREADS 40
#define BLOCKS  9
#define TOTAL   (THREADS * BLOCKS)

uint32_t out_block[TOTAL], out_thread[TOTAL], out_bdim[TOTAL], out_gdim[TOTAL], out_hw[TOTAL];

void identity(void* unused) {
    (void)unused;
    uint32_t i = simt_global_id();
    out_block[i]  = simt_block_idx();
    out_thread[i] = simt_thread_idx();
    out_bdim[i]   = simt_block_dim();
    out_gdim[i]   = simt_grid_dim();
    out_hw[i]     = simt_hw_tid();
}

int main(void) {
    simt_launch(identity, THREADS, BLOCKS, 0);
    return 0;
}
