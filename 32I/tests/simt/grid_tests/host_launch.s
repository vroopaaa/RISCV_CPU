# host_launch -- the host CPU launches `kernel` (link with identity.s) over
# NB blocks x TPB threads (set with --defsym), then writes a marker to prove
# it resumed after the launch.
    .include "simt_macros.inc"
    .equ MARKER_ADDR, 0x180000
    .text
    .globl _start
    .globl _end
_start:
    la   sp, _stack_top
    li   a0, 0x100000              # kernel arg: output base
    la   t0, kernel
    li   t1, (NB << 16) | TPB
    SIMT_LAUNCH t0, t1
    li   t2, 0x600D
    li   t3, MARKER_ADDR
    sw   t2, 0(t3)                 # "host ran after the launch"
_end:
    j    _end
