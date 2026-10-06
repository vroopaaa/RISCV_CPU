# divergent -- if/else plus a nested if, all with SPLIT/JOIN:
#   if (threadIdx odd) { x = gid*3; if (threadIdx < 16) x += 1000; }
#   else               { x = gid + 100; }
#   a0 = out base, out[gid] = x.
    .include "simt_macros.inc"
    .text
    .globl kernel
kernel:
    SIMT_BLOCK_IDX  t0
    SIMT_BLOCK_DIM  t1
    SIMT_THREAD_IDX t3
    mul  t4, t0, t1
    add  t4, t4, t3          # gid
    li   a1, 0               # x
    andi t5, t3, 1           # odd?
    SIMT_SPLIT t5, zero      # then: odd lanes
    slli a1, t4, 1
    add  a1, a1, t4          # x = gid*3
    sltiu t6, t3, 16
    SIMT_SPLIT t6, zero      # nested then: threadIdx < 16
    addi a1, a1, 1000
    SIMT_JOIN
    SIMT_JOIN
    xori t5, t5, 1           # even?
    SIMT_SPLIT t5, zero      # else: even lanes
    addi a1, t4, 100         # x = gid + 100
    SIMT_JOIN
    slli t4, t4, 2
    add  t4, a0, t4
    sw   a1, 0(t4)
    ret
