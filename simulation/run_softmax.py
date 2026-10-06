"""
Driver for the softmax kernel (kernels.SOFTMAX_KERNEL) -- a fixed-point,
lookup-table-based softmax over a 4-element vector, one element per lane.

Expected output is computed independently here using the SAME fixed-point
LUT + integer-division arithmetic the kernel itself does (not true
floating-point softmax) -- the LUT quantization and integer division are
part of the design, not precision bugs to paper over, so the diff should be
bit-exact, same philosophy as 32I/tests/npu/run_npu_tests.py's approach of
computing an independent expected result rather than trusting the core.
"""

from simt_model import SIMTCore
from kernels import (
    SOFTMAX_KERNEL,
    SOFTMAX_LUT,
    SOFTMAX_LUT_BASE,
    SOFTMAX_LUT_MIN,
    SOFTMAX_LUT_MAX,
    SOFTMAX_OUT_BASE,
    SOFTMAX_OUT_SCALE,
    SOFTMAX_SHARED_MEM_SIZE,
    SOFTMAX_VEC_LEN,
)


def compute_expected(x):
    """Mirrors the kernel's own fixed-point arithmetic exactly, including
    integer-division truncation, so this is a bit-exact reference, not an
    approximation of true softmax."""
    exps = [SOFTMAX_LUT[xi - SOFTMAX_LUT_MIN] for xi in x]
    total = sum(exps)
    return [(e * SOFTMAX_OUT_SCALE) // total for e in exps]  # truncating division, all operands non-negative


def main():
    x = [2, 0, -1, 1]
    assert all(SOFTMAX_LUT_MIN <= xi <= SOFTMAX_LUT_MAX for xi in x), "x must be in the LUT's domain"
    expected = compute_expected(x)

    core = SIMTCore(shared_mem_size=SOFTMAX_SHARED_MEM_SIZE)
    core.shared_mem[0:SOFTMAX_VEC_LEN] = x
    core.shared_mem[SOFTMAX_LUT_BASE:SOFTMAX_LUT_BASE + len(SOFTMAX_LUT)] = SOFTMAX_LUT
    core.load_kernel(SOFTMAX_KERNEL)
    cycles = core.run(max_cycles=200)

    actual = core.shared_mem[SOFTMAX_OUT_BASE:SOFTMAX_OUT_BASE + SOFTMAX_VEC_LEN]

    print(f"x:        {x}")
    print(f"LUT:      {SOFTMAX_LUT}  (domain {SOFTMAX_LUT_MIN}..{SOFTMAX_LUT_MAX}, exp(x)*{SOFTMAX_OUT_SCALE})")
    print(f"ran {cycles} cycles")
    print(f"expected (per-mille): {expected}  (sum={sum(expected)})")
    print(f"actual   (per-mille): {actual}  (sum={sum(actual)})")
    print("PASS" if actual == expected else "FAIL")


if __name__ == "__main__":
    main()
