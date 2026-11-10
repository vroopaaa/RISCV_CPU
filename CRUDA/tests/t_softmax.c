// t_softmax -- one nn_softmax call on the tensors run_nn_tests.py generated
// into data.bin (offsets and sizes in data.h). Returns nn_softmax's result in
// a0; the driver then checks a0 and dumps the output buffer.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const uint16_t* table = HAS_TABLE ? (const uint16_t*)(nn_data + TABLE_OFF) : 0;
    return nn_softmax((const int8_t*)(nn_data + X_OFF), LDT_X,
                      (int8_t*)(nn_data + Y_OFF), LDT_Y,
                      SM_ROWS, SM_COLS,
                      table);
}
