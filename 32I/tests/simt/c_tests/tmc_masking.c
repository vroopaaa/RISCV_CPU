// TMC masking: narrow to 2 active threads, confirm only those lanes write
// memory -- lanes outside the narrowed mask must leave their destination
// word untouched (checked against a sentinel by the Python driver).
#include <stdint.h>
#include "../simt_isa.h"

#define ACTIVE_COUNT 2
#define RESULT_BASE 0x100000

int main(void) {
    uint32_t idx = simt_tid() & 0xFF;
    int32_t* out = (int32_t*)RESULT_BASE;

    // Seed a sentinel with ALL lanes still active, so narrowing is the only
    // thing that decides who overwrites it below -- no harness-side
    // pre-seeding needed.
    out[idx] = 0xDEADBEEF;

    simt_tmc(ACTIVE_COUNT);
    idx = simt_tid() & 0xFF; // re-read: still this lane's own idx, just re-fetched post-TMC
    out[idx] = 7;
    return 0;
}
