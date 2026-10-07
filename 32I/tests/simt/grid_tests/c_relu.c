// c_relu -- leaky ReLU across blocks: out = x >= 0 ? x : x / 8 (C truncating
// division). A divergent if/else, written as two SPLIT..JOIN regions.
#include <stdint.h>
#include "simt_isa.h"

#define THREADS 32
#define BLOCKS  7
#define N       (THREADS * BLOCKS)

struct relu_args { const int32_t* in; int32_t* out; };

int32_t relu_in[N], relu_out[N];
struct relu_args args;

void leaky_relu(void* p) {
    struct relu_args* r = (struct relu_args*)p;
    uint32_t i = simt_global_id();
    int32_t v = r->in[i];
    uint32_t neg = v < 0;
    if (simt_split(neg))           // if (v < 0)
        r->out[i] = v / 8;
    simt_join();
    if (simt_split(!neg))          // else
        r->out[i] = v;
    simt_join();
}

int main(void) {
    for (int32_t i = 0; i < N; i++) relu_in[i] = (i * 37) % 201 - 100;   // -100..100, mixed signs per warp
    args.in = relu_in; args.out = relu_out;
    simt_launch(leaky_relu, THREADS, BLOCKS, &args);
    return 0;
}
