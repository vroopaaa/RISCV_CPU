// Tests loops + branches + array/memory access
int compute() {
    int a[10] = {5, 2, 9, 1, 5, 6, 7, 3, 8, 4};
    int n = sizeof(a) / sizeof(a[0]);
    int i, j, temp;

    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                temp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = temp;
            }
        }
    }
    // Check whether the array ended up sorted.
    int is_sorted = 1;
    for (i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) {
            is_sorted = 0;
            break;
        }
    }

    return is_sorted;   // expected: 1
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
