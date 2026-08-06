// Tests loops + branches
int fib(int n) {
    volatile int a = 0;
    volatile int b = 1;
    volatile int i;

    for (i = 0; i < n; i++) {
        volatile int temp = a + b;
        a = b;
        b = temp;
    }

    return a;
}

int compute() {
    return fib(10);   // expected: 55
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
