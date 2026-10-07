// c_bad_launch -- two launches the GPU must refuse (129 threads per block, and
// 0 blocks). Both are skipped, the host carries on and sets `marker`.
#include <stdint.h>
#include "simt_isa.h"

uint32_t out[256];
uint32_t marker;

void touch(void* unused) {
    (void)unused;
    out[simt_global_id()] = 1;
}

int main(void) {
    simt_launch(touch, 129, 1, 0);
    simt_launch(touch, 32, 0, 0);
    marker = 0x600D;
    return 0;
}
