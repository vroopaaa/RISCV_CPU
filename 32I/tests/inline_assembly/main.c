// Multiply two operand using custom inline assembly for risc V

static inline int custom_mul(int a, int b){
    int result;
    asm volatile(".insn r 0x0B, 0, 0, %0, %1, %2" : "=r"(result) : "r"(a), "r"(b));
    return result;
}

int main(){
    volatile int a = 5;
    volatile int b = 10;
    volatile int result = custom_mul(a, b);
    return result;
}