"""
TEMPORARY visualization helper -- prints what every lane is doing, cycle by
cycle. Not part of the core model (doesn't touch simt_model.py or
kernels.py), so delete this file whenever you don't need the visibility
anymore; nothing else depends on it.

Usage:
    from trace import trace_run
    core = SIMTCore(shared_mem_size=...)
    core.shared_mem[...] = ...
    core.load_kernel(SOME_KERNEL)
    trace_run(core)                       # all registers
    trace_run(core, show_regs=[0, 3, 4])  # narrow the register columns
"""

from simt_model import THREADS_PER_WARP, NUM_REGS, WARP_LEVEL_OPS


def trace_run(core, max_cycles: int = 1000, show_regs=None) -> int:
    """Runs core cycle by cycle (same loop as SIMTCore.run()), printing one
    block per cycle: the instruction and active-lane mask it ran with, then
    one row per lane showing whether it was active and its register state
    AFTER the instruction executed (so you see the effect, not just the
    inputs). Returns the number of cycles run, same as core.run()."""
    if show_regs is None:
        show_regs = list(range(NUM_REGS))

    col_header = " ".join(f"r{r:>2}" for r in show_regs)
    print(f"{'cyc':>4} {'pc':>4}  {'instr':<24} {'mask':>6}  {'ipdom':>5}   lane | {col_header}")
    print("-" * (26 + 13 + len(col_header) + 20))

    while not core.warp.halted and core.cycle < max_cycles:
        cyc = core.cycle
        pc = core.warp.pc
        instr, *args = core.kernel[pc]
        mask_before = core.warp.tmask           # mask the instruction actually ran under
        mask_str = format(mask_before, f"0{THREADS_PER_WARP}b")
        depth_before = len(core.warp.ipdom_stack)
        instr_str = f"{instr}{tuple(args)}" if args else f"{instr}()"
        level_tag = "[warp]" if instr in WARP_LEVEL_OPS else "[lane]"

        core.step()  # actually executes -- may change tmask/pc/regs/ipdom_stack

        depth_after = len(core.warp.ipdom_stack)
        depth_str = f"{depth_before}" if depth_after == depth_before else f"{depth_before}->{depth_after}"
        print(f"{cyc:>4} {pc:>4}  {instr_str:<17} {level_tag} {mask_str:>6}  {depth_str:>5}")

        for lane in range(THREADS_PER_WARP):
            active = bool(mask_before & (1 << lane))
            marker = "*" if active else " "
            regs = " ".join(f"{core.warp.regs[lane][r]:>3}" for r in show_regs)
            print(f"                                               lane{lane}{marker} | {regs}")

    status = "HALTED" if core.warp.halted else f"STOPPED (max_cycles={max_cycles} reached)"
    print(f"\n{status} after {core.cycle} cycles")
    return core.cycle


# ---------------------------------------------------------------------
# CLI: `python3 trace.py [kernel_name]` -- runs one of the kernels from
# kernels.py through trace_run() with sensible default shared_mem/inputs,
# so you don't need to write a driver script just to look at a trace.
# ---------------------------------------------------------------------

def _demo_scalar_multiply():
    from kernels import SCALAR_MULTIPLY_KERNEL, SCALAR_MULTIPLY_SCALAR_VALUE
    from simt_model import SIMTCore
    vec = [2, 3, 5, 7]
    core = SIMTCore(shared_mem_size=len(vec))
    core.shared_mem[:] = vec
    core.load_kernel(SCALAR_MULTIPLY_KERNEL)
    print(f"vec={vec}  scalar={SCALAR_MULTIPLY_SCALAR_VALUE}\n")
    return core, dict(show_regs=[0, 2, 3, 4])


def _demo_divergent_branch():
    from kernels import DIVERGENT_BRANCH_KERNEL
    from simt_model import SIMTCore, THREADS_PER_WARP
    core = SIMTCore(shared_mem_size=THREADS_PER_WARP)
    core.load_kernel(DIVERGENT_BRANCH_KERNEL)
    return core, dict(show_regs=[0, 5, 6, 7])


def _demo_leaky_relu():
    from kernels import LEAKY_RELU_KERNEL, LEAKY_RELU_OUT_BASE
    from simt_model import SIMTCore
    x = [5, -3, 8, -10]
    core = SIMTCore(shared_mem_size=LEAKY_RELU_OUT_BASE + len(x))
    core.shared_mem[0:len(x)] = x
    core.load_kernel(LEAKY_RELU_KERNEL)
    print(f"x={x}\n")
    return core, dict(show_regs=[0, 1, 3, 4])


def _demo_softmax():
    from kernels import (
        SOFTMAX_KERNEL, SOFTMAX_LUT, SOFTMAX_LUT_BASE, SOFTMAX_SHARED_MEM_SIZE, SOFTMAX_VEC_LEN,
    )
    from simt_model import SIMTCore
    x = [2, 0, -1, 1]
    core = SIMTCore(shared_mem_size=SOFTMAX_SHARED_MEM_SIZE)
    core.shared_mem[0:SOFTMAX_VEC_LEN] = x
    core.shared_mem[SOFTMAX_LUT_BASE:SOFTMAX_LUT_BASE + len(SOFTMAX_LUT)] = SOFTMAX_LUT
    core.load_kernel(SOFTMAX_KERNEL)
    print(f"x={x}\n")
    return core, dict(show_regs=[0, 1, 3, 5])


_DEMOS = {
    "scalar_multiply": _demo_scalar_multiply,
    "divergent_branch": _demo_divergent_branch,
    "leaky_relu": _demo_leaky_relu,
    "softmax": _demo_softmax,
}

if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "leaky_relu"
    if name not in _DEMOS:
        print(f"Unknown kernel '{name}'. Choices: {', '.join(_DEMOS)}")
        sys.exit(1)

    core, trace_kwargs = _DEMOS[name]()
    trace_run(core, **trace_kwargs)
