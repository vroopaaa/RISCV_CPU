// t_lut8 -- one nn_lut8 call on the tensors run_nn_tests.py generated into
// data.bin (offsets and sizes in data.h). Returns nn_lut8's result in a0;
// the driver then checks a0 and dumps the output buffer.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const int8_t* table = HAS_TABLE ? (const int8_t*)(nn_data + TABLE_OFF) : 0;
    return nn_lut8((const int8_t*)(nn_data + X_OFF), LDT_X,
                   (int8_t*)(nn_data + Y_OFF), LDT_Y,
                   LUT_ROWS, LUT_COLS,
                   table);
}
