"""
Driver for the Leaky ReLU kernel (kernels.LEAKY_RELU_KERNEL) -- see that
kernel's header comment for why it's a more realistic divergence example
than DIVERGENT_BRANCH_KERNEL: both branches compute something genuinely
different from the lane's own data, instead of just loading a constant.
"""

from simt_model import SIMTCore
from kernels import LEAKY_RELU_KERNEL, LEAKY_RELU_OUT_BASE


def to_signed32(v: int) -> int:
    """shared_mem stores raw unsigned 32-bit words (two's complement for
    negatives), same as every register write in simt_model.py -- a result
    like -1 is stored as 4294967295. Interpret it back as signed before
    comparing/printing, same convention as simt_model._to_signed32."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def div_trunc(a: int, b: int) -> int:
    """Same truncating-toward-zero division as simt_model._op_div, so the
    expected values are computed with identical semantics, not just
    "close enough" floating-point division."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def compute_expected(x):
    return [xi if xi >= 0 else div_trunc(xi, 8) for xi in x]


def main():
    x = [5, -3, 8, -10]  # mixed signs -> lanes 0,2 take one branch, 1,3 the other
    expected = compute_expected(x)

    core = SIMTCore(shared_mem_size=LEAKY_RELU_OUT_BASE + len(x))
    core.shared_mem[0:len(x)] = x
    core.load_kernel(LEAKY_RELU_KERNEL)
    cycles = core.run(max_cycles=100)

    actual = [to_signed32(v) for v in core.shared_mem[LEAKY_RELU_OUT_BASE:LEAKY_RELU_OUT_BASE + len(x)]]

    print(f"x:        {x}")
    print(f"ran {cycles} cycles")
    print(f"expected: {expected}")
    print(f"actual:   {actual}")
    print("PASS" if actual == expected else "FAIL")
    print(f"ipdom_stack after run (should be []): {core.warp.ipdom_stack}")


if __name__ == "__main__":
    main()
