# stack -- every lane uses its private stack: spills to its own frame, calls a
# helper that builds a second frame, then reloads its spills.
#   a0 = out base. out[gid] = gid*gid + gid + 1, out[total + gid] = sp at entry.
# If two lanes shared a stack slot, their spills would overwrite each other
# (all lanes of a warp store in the same instruction) and the sums go wrong.
    .include "simt_macros.inc"
    .text
    .globl kernel
kernel:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s0, 8(sp)
    mv   s0, a0              # out base, callee-saved across the call
    SIMT_BLOCK_IDX  t0
    SIMT_BLOCK_DIM  t1
    SIMT_THREAD_IDX t3
    mul  t5, t0, t1
    add  t5, t5, t3          # gid
    addi t6, sp, 16          # sp at entry
    sw   t5, 4(sp)           # spill gid
    sw   t6, 0(sp)           # spill entry sp
    mv   a0, t5
    call square_plus1        # a0 = gid*gid + 1
    lw   t5, 4(sp)           # reload spills
    lw   t6, 0(sp)
    add  a0, a0, t5          # gid*gid + gid + 1
    SIMT_BLOCK_DIM t1
    SIMT_GRID_DIM  t2
    mul  t2, t2, t1
    slli t2, t2, 2           # bytes per field
    slli t5, t5, 2
    add  t5, s0, t5          # &out[gid]
    sw   a0, 0(t5)
    add  t5, t5, t2
    sw   t6, 0(t5)           # out[total + gid] = entry sp
    lw   s0, 8(sp)
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret

square_plus1:                # a0 = a0*a0 + 1, through its own stack frame
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   a0, 8(sp)
    lw   t0, 8(sp)
    mul  t0, t0, t0
    addi a0, t0, 1
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret
