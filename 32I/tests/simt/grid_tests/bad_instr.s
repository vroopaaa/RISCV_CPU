# bad_instr -- every thread writes 1 to out[gid], then executes BAD_WORD (set
# with --defsym: an instruction the GPU can't run), then would write 2.
# The first warp to reach BAD_WORD must fault: the block stops there, the
# rest of the grid is abandoned, and nothing ever writes 2.
# With BAD_WORD = a FENCE (a legal no-op on the GPU), every thread ends at 2.
#   a0 = out base
    .include "simt_macros.inc"
    .text
    .globl kernel
    .globl bad_at
kernel:
    SIMT_BLOCK_IDX  t0
    SIMT_BLOCK_DIM  t1
    SIMT_THREAD_IDX t3
    mul  t4, t0, t1
    add  t4, t4, t3          # gid
    slli t4, t4, 2
    add  t4, a0, t4          # &out[gid]
    li   t5, 1
    sw   t5, 0(t4)           # before the bad instruction
bad_at:
    .word BAD_WORD
    li   t5, 2
    sw   t5, 0(t4)           # after it -- must never run if BAD_WORD faults
    ret
