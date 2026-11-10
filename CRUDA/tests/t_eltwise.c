// t_eltwise -- one nn_add (ELT_OP 0) or nn_mul (ELT_OP 1) call on the
// tensors run_nn_tests.py generated into data.bin (offsets and sizes in
// data.h). Returns the wrapper's result in a0; the driver then checks a0 and
// dumps the output buffer.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const int8_t* A = (const int8_t*)(nn_data + A_OFF);
    const int8_t* B = HAS_B ? (const int8_t*)(nn_data + B_OFF) : 0;
    int8_t*       Y = (int8_t*)(nn_data + Y_OFF);
#if ELT_OP == 0
    return nn_add(A, LDT_A, ELT_SA, B, LDT_B, ELT_SB, Y, LDT_Y, ELT_ROWS, ELT_COLS, ELT_SHIFT);
#else
    return nn_mul(A, LDT_A, B, LDT_B, Y, LDT_Y, ELT_ROWS, ELT_COLS, ELT_SHIFT);
#endif
}
