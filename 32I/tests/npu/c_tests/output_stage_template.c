// Template for the NPU output-stage test (run_npu_tests.py --output-stage).
// run_npu_tests.py fills in the NUM_CASES and CASES placeholders below with
// random cases and checks out[] against its own Python model; this file is
// never compiled as-is.
//
// Every case sets DIM_M = m, DIM_K = 16, DIM_N = n and owns 1024 bytes of
// out[] (room for one int32 tile or up to four int8 tiles). What it does
// depends on `mode`:
//   0: load_bias (C = bias), MAC, store_c8                  -> int8 tile 0
//   1: the same through MMIO only: bias words written to the broadcast
//      port, shift written to its register, OUT8 read with plain loads
//   2: RESET clears the bias: load_bias, MAC, store_c8 -> tile 0; then
//      RESET, MAC, store_c8 -> tile 1 (no bias this time)
//   3: MACs accumulate on top of the bias: load_bias, MAC, MAC, store_c8
//   4: load_bias alone, then store_c (int32, 64-byte rows): C == bias in
//      every row
#include <stdint.h>
#include "npu_isa.h"

struct os_case {
    int8_t   a[NPU_TILE_ELEMS];
    int8_t   b[NPU_TILE_ELEMS];
    int32_t  bias[NPU_TILE_DIM];
    uint32_t m, n, shift, mode;
};

#define NUM_CASES {{NUM_CASES}}

static const struct os_case cases[NUM_CASES] __attribute__((aligned(4))) = {
{{CASES}}
};

int8_t out[NUM_CASES][4 * NPU_TILE_ELEMS] __attribute__((aligned(4)));

static void mac_tile(const struct os_case* t) {
    npu_load_a(t->a);
    npu_load_b(t->b);
    npu_write32(NPU_MAC, 1);
}

int main(void) {
    npu_write32(NPU_DIM_K, NPU_TILE_DIM);
    for (uint32_t c = 0; c < NUM_CASES; c++) {
        const struct os_case* t = &cases[c];
        int8_t* o = out[c];
        npu_write32(NPU_DIM_M, t->m);
        npu_write32(NPU_DIM_N, t->n);
        if (t->mode == 1) {
            for (uint32_t j = 0; j < NPU_TILE_DIM; j++)
                npu_write32(NPU_BIAS + j * 4, (uint32_t)t->bias[j]);
            mac_tile(t);
            npu_write32(NPU_OUT_SHIFT, t->shift);
            uint32_t* dst = (uint32_t*)o;
            for (uint32_t w = 0; w < NPU_TILE_ELEMS / 4; w++)
                dst[w] = npu_read32(NPU_OUT8 + w * 4);
        } else if (t->mode == 4) {
            npu_load_bias(t->bias);
            npu_store_c((int32_t*)o, NPU_TILE_DIM * 4);
        } else {
            npu_load_bias(t->bias);
            mac_tile(t);
            if (t->mode == 3)
                mac_tile(t);
            npu_store_c8(o, t->shift);
            if (t->mode == 2) {
                npu_write32(NPU_RESET, 1);
                mac_tile(t);
                npu_store_c8(o + NPU_TILE_ELEMS, t->shift);
            }
        }
    }
    return 0;
}
