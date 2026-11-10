// t_cycles -- nn_cycles() across the 2^32 wrap. run_nn_tests.py presets the
// counter to CYCLE_START, just below 2^32, so the low word wraps while this
// loop samples. A torn read (high word from before the wrap, low word from
// after, or the reverse) shows up as a sample that goes backwards or jumps by
// about 2^32. Returns 0 if every sample is after the previous one by less
// than 1000 cycles; 1 if not; 2 if the first sample is before CYCLE_START or
// the counter never passed 2^32.
#include <stdint.h>
#include "nn.h"
#include "data.h"

int main(void) {
    uint64_t first = nn_cycles();
    uint64_t prev = first;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < CYCLE_SAMPLES; i++) {
        uint64_t t = nn_cycles();
        bad |= (t <= prev) | (t - prev >= 1000u);
        prev = t;
    }
    if (bad)
        return 1;
    if (first < CYCLE_START || (prev >> 32) != 1u)
        return 2;
    return 0;
}
