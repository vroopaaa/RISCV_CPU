# wrap -- run with the counter preset to 2^32 - 3 (run_test's start-cycle
# argument). The first reads see the low word just below the wrap and a high
# word of 0; the last reads, after the wrap, see a high word of 1 and the
# low word of the final count. The jump to 1: starts a fresh fetch window,
# as in total.s.
    .text
    .globl _start
    .globl _end
_start:
    rdcycleh a2
    rdcycle  a3
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    j    1f
1:  rdcycleh a1
    rdcycle  a0
_end:
    j    _end
