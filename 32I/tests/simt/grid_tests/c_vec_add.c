// c_vec_add -- CUDA-style vector add in plain C. The host fills a/b, launches
// vec_add over BLOCKS x THREADS, then checks c itself and returns the number of
// correct elements (so the same binary also works under emul: a0 == 200).
#include <stdint.h>
#include "simt_isa.h"

#define N       190
#define THREADS 40
#define BLOCKS  5                 // 200 threads: the last 10 are out of range
#define TOTAL   (THREADS * BLOCKS)
#define POISON  0xDEADBEEFu

struct vec_args { const uint32_t* a; const uint32_t* b; uint32_t* c; uint32_t n; };

uint32_t vec_a[TOTAL], vec_b[TOTAL], vec_c[TOTAL];
struct vec_args args;

void vec_add(void* p) {
    struct vec_args* v = (struct vec_args*)p;
    uint32_t i = simt_global_id();
    uint32_t in_range = i < v->n;
    if (simt_split(in_range))      // only in-range lanes run until simt_join
        v->c[i] = v->a[i] + v->b[i];
    simt_join();
}

int main(void) {
    for (uint32_t i = 0; i < TOTAL; i++) {
        vec_a[i] = i * 3 + 1;
        vec_b[i] = 1000 - i * 7;
        vec_c[i] = POISON;
    }
    args.a = vec_a; args.b = vec_b; args.c = vec_c; args.n = N;

    simt_launch(vec_add, THREADS, BLOCKS, &args);

    uint32_t correct = 0;
    for (uint32_t i = 0; i < TOTAL; i++) {
        uint32_t want = (i < N) ? vec_a[i] + vec_b[i] : POISON;
        if (vec_c[i] == want) correct++;
    }
    return correct;
}
