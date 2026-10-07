// c_two_launches -- launch, host work, launch again, host work:
//   y = 2x (GPU); y += 1 (host); z = 2y (GPU); result = sum(z) (host)
// The host must see the GPU's writes and resume correctly after each launch.
#include <stdint.h>
#include "simt_isa.h"

#define THREADS 64
#define BLOCKS  6
#define N       (THREADS * BLOCKS)

struct scale_args { const int32_t* in; int32_t* out; uint32_t n; };

int32_t x[N], y[N], z[N];
int32_t result;
struct scale_args args;

void scale2(void* p) {
    struct scale_args* s = (struct scale_args*)p;
    uint32_t i = simt_global_id();
    uint32_t in_range = i < s->n;
    if (simt_split(in_range))
        s->out[i] = s->in[i] * 2;
    simt_join();
}

int main(void) {
    for (int32_t i = 0; i < N; i++) x[i] = i * 5 - 100;

    args.in = x; args.out = y; args.n = N;
    simt_launch(scale2, THREADS, BLOCKS, &args);

    for (int32_t i = 0; i < N; i++) y[i] += 1;

    args.in = y; args.out = z;
    simt_launch(scale2, THREADS, BLOCKS, &args);

    int32_t sum = 0;
    for (int32_t i = 0; i < N; i++) sum += z[i];
    result = sum;
    return 0;
}
