// t_isqrt -- nn_isqrt64 on every value run_nn_tests.py put in data.bin
// (ISQRT_N values, each as two words: high, low). Writes one result per value
// and returns ISQRT_N in a0; the driver compares against math.isqrt.
#include <stdint.h>
#include "nn.h"
#include "data.h"

extern int8_t nn_data[];

int main(void) {
    const uint32_t* v = (const uint32_t*)(nn_data + V_OFF);
    uint32_t*       q = (uint32_t*)(nn_data + Q_OFF);
    for (uint32_t i = 0; i < ISQRT_N; i++)
        q[i] = nn_isqrt64(v[2 * i], v[2 * i + 1]);
    return ISQRT_N;
}
