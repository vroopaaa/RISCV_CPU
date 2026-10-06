"""
Driver for the scalar-multiply kernel: plays the role the C++ test harness
(host side) will eventually play -- seed shared memory, "launch" the
kernel on the warp, run to completion, check the result.

Mirrors the NPU test harness pattern already in the repo
(32I/tests/npu/run_npu_tests.py): Python computes the expected result
independently and diffs it against what the simulated core produced,
instead of trusting the core's own output.
"""

from simt_model import SIMTCore, THREADS_PER_WARP
from kernels import SCALAR_MULTIPLY_KERNEL, SCALAR_MULTIPLY_SCALAR_VALUE


def main():
    vec = [2, 3, 5, 7][:THREADS_PER_WARP]
    scalar = SCALAR_MULTIPLY_SCALAR_VALUE
    expected = [v * scalar for v in vec]

    core = SIMTCore(shared_mem_size=len(vec))
    core.shared_mem[:] = vec
    core.load_kernel(SCALAR_MULTIPLY_KERNEL)
    cycles = core.run(max_cycles=100)

    actual = core.shared_mem
    print(f"ran {cycles} cycles")
    print(f"expected: {expected}")
    print(f"actual:   {actual}")
    print("PASS" if actual == expected else "FAIL")

    # Masking: only lanes 0 and 2 active (tmask = 0b0101) -- lanes 1 and 3
    # should leave their shared_mem slot untouched.
    core2 = SIMTCore(shared_mem_size=len(vec))
    core2.shared_mem[:] = vec
    core2.load_kernel(SCALAR_MULTIPLY_KERNEL, tmask=0b0101)
    core2.run(max_cycles=100)
    masked_expected = [vec[0] * scalar, vec[1], vec[2] * scalar, vec[3]]
    print(f"\nmasked (tmask=0b0101) expected: {masked_expected}")
    print(f"masked (tmask=0b0101) actual:   {core2.shared_mem}")
    print("PASS" if core2.shared_mem == masked_expected else "FAIL")


if __name__ == "__main__":
    main()
