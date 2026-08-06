// Recursive factorial - stresses jal/jalr and stack save/restore
// via compiler-generated function calls, similar to your original
// hand-written test but now compiler-generated.
int fact(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}

int main() {
    volatile int result = fact(5);   // 5! = 120
    return result;
}