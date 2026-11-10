# same_cycle -- two reads back to back, a0 = second - first. The scalar host
# runs them in consecutive cycles (1); the superscalar host issues both in one
# group, so they read the same cycle (0).
    .text
    .globl _start
    .globl _end
_start:
    rdcycle  t0
    rdcycle  t1
    sub      a0, t1, t0
_end:
    j    _end
