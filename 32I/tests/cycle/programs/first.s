# first -- rdcycle as the very first instruction: no cycle has completed yet,
# so both halves read 0.
    .text
    .globl _start
    .globl _end
_start:
    rdcycle  a0
    rdcycleh a1
_end:
    j    _end
