# launch -- the GPU's device cycles are part of the host's count. One block
# of 64 threads (2 warps on SM 0) runs identity.s's straight-line kernel;
# a0 = read after - read before = device cycles + 2 (the first read's own
# cycle and the LAUNCH's cycle). LAUNCH issues alone on the superscalar host,
# so the reads are never in its group.
    .include "simt_macros.inc"
    .text
    .globl _start
    .globl _end
_start:
    li   a0, 0x100000              # kernel arg: output base
    la   t0, kernel
    li   t1, (1 << 16) | 64
    rdcycle s0
    SIMT_LAUNCH t0, t1
    rdcycle s1
    sub  a0, s1, s0
_end:
    j    _end
