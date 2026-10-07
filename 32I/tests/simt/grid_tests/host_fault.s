# host_fault -- the host launches `kernel` (link with bad_instr.s or
# runaway.s), which fails on the GPU. The CPU must see the GPU's flag when the
# launch returns and halt right there: the marker after the launch is never
# written.
    .include "simt_macros.inc"
    .equ MARKER_ADDR, 0x180000
    .text
    .globl _start
    .globl launch_at
_start:
    la   sp, _stack_top
    li   a0, 0x100000
    la   t0, kernel
    li   t1, (NB << 16) | TPB
    li   t2, 0x600D
    li   t3, MARKER_ADDR
launch_at:
    SIMT_LAUNCH t0, t1
    sw   t2, 0(t3)           # must never run -- and, with its operands ready, it is the
                             # instruction a superscalar CPU would co-issue with the LAUNCH
_end:
    j    _end
