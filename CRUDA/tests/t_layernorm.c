// t_layernorm -- one nn_layernorm call on the tensors run_nn_tests.py
// generated into data.bin (offsets and sizes in data.h). Returns
// nn_layernorm's result in a0; the driver then checks a0 and dumps the
// output buffer.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const int8_t*  gamma = HAS_GAMMA ? (const int8_t*)(nn_data + GAMMA_OFF) : 0;
    const int32_t* beta  = HAS_BETA ? (const int32_t*)(nn_data + BETA_OFF) : 0;
    return nn_layernorm((const int8_t*)(nn_data + X_OFF), LDT_X,
                        (int8_t*)(nn_data + Y_OFF), LDT_Y,
                        LN_ROWS, LN_COLS,
                        gamma, beta, LN_SHIFT);
}
