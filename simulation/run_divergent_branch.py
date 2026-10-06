"""
Driver for the manufactured divergent-branch kernel (kernels.DIVERGENT_BRANCH_KERNEL)
-- the one hand-written test SPLIT/JOIN need to prove the IPDOM stack mechanism
is correct, per simulation/README.md.
"""

from simt_model import SIMTCore, THREADS_PER_WARP
from kernels import DIVERGENT_BRANCH_KERNEL


def main():
    expected = [100, 100, 200, 200]  # lanes 0,1 take idx<2; lanes 2,3 take idx>=2

    core = SIMTCore(shared_mem_size=THREADS_PER_WARP)
    core.load_kernel(DIVERGENT_BRANCH_KERNEL)
    cycles = core.run(max_cycles=100)

    print(f"ran {cycles} cycles")
    print(f"expected: {expected}")
    print(f"actual:   {core.shared_mem}")
    print("PASS" if core.shared_mem == expected else "FAIL")

    # ipdom_stack should be back to empty -- every SPLIT was matched by a JOIN.
    print(f"ipdom_stack after run (should be []): {core.warp.ipdom_stack}")


if __name__ == "__main__":
    main()
