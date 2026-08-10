// Tests recursion + stack save/restore (jal/jalr) + MUL
int fact(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}

int compute() {
    // volatile: without this, GCC constant-folds the whole recursion at -O2+
    // into `li a0,120; ret` -- no jal/jalr/MUL would run at all.
    volatile int n = 5;
    return fact(n);   // expected: 120
}

#ifdef NATIVE_TEST
#include <stdio.h>
int main() {
    printf("%d\n", compute());
    return 0;
}
#else
int main() {
    volatile int result = compute();
    return result;
}
#endif
