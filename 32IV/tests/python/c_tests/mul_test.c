// Tests basic arithmetic + M-extension MUL
int compute() {
    volatile int a = 5;
    volatile int b = 7;
    volatile int result = a * b;
    return result;   // expected: 35
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
