# total -- a loop with loads, stores and branches, then one read just before
# the halt. The jump to 1: makes the read and the halting jump start a fresh
# fetch window, so on the superscalar host they issue together. a0 must be
# the harness's final cycle count minus the cycles of those last two
# instructions (2 scalar, 1 superscalar width >= 2).
    .text
    .globl _start
    .globl _end
_start:
    li   t0, 37
    li   t1, 0x100000
loop:
    lw   t2, 0(t1)
    addi t2, t2, 1
    sw   t2, 0(t1)
    addi t0, t0, -1
    bnez t0, loop
    j    1f
1:  rdcycle a0
_end:
    j    _end
