// t_gemm -- one nn_gemm call on the tensors run_nn_tests.py generated into
// data.bin (offsets and sizes in data.h). Returns nn_gemm's result in a0;
// the driver then checks a0 and dumps the output buffer.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const int32_t* bias = HAS_BIAS ? (const int32_t*)(nn_data + BIAS_OFF) : 0;
    return nn_gemm((const int8_t*)(nn_data + X_OFF), LDT_X,
                   (const int8_t*)(nn_data + W_OFF), LDT_W,
                   bias,
                   (int8_t*)(nn_data + Y_OFF), LDT_Y,
                   GEMM_M, GEMM_K, GEMM_N, GEMM_SHIFT);
}
