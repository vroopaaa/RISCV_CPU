# host_two_launches -- link with vec_add.s and the generated data file.
#   launch 1: c = a + b              (args = vec_args  = {a, b, c, n})
#   launch 2: d = c + b = a + 2b     (args = vec_args2 = {c, b, d, n})
#   then the host itself sums d[0..n) into host_sum -- it must see the
#   device's writes and resume at the right pc after each launch.
    .include "simt_macros.inc"
    .equ MARKER_ADDR, 0x180000
    .text
    .globl _start
    .globl _end
_start:
    la   sp, _stack_top
    la   s0, kernel
    li   s1, (NB << 16) | TPB
    la   a0, vec_args
    SIMT_LAUNCH s0, s1
    la   a0, vec_args2
    SIMT_LAUNCH s0, s1
    la   t0, vec_args2
    lw   t1, 8(t0)                 # d
    lw   t2, 12(t0)                # n
    li   t3, 0                     # sum
sum_loop:
    beqz t2, sum_done
    lw   t4, 0(t1)
    add  t3, t3, t4
    addi t1, t1, 4
    addi t2, t2, -1
    j    sum_loop
sum_done:
    la   t0, host_sum
    sw   t3, 0(t0)
    li   t2, 0x600D
    li   t3, MARKER_ADDR
    sw   t2, 0(t3)
_end:
    j    _end
