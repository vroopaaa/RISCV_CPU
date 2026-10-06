# identity -- every thread records who it is.
#   a0 = out base. total = gridDim*blockDim threads, field f at out[f*total + gid]:
#   f0 blockIdx, f1 blockDim, f2 gridDim, f3 threadIdx, f4 hw_tid
# Straight-line on purpose (no branches): run_grid_tests.py uses
# (kernel_end - kernel)/4 as the per-warp instruction count to check cycles.
    .include "simt_macros.inc"
    .text
    .globl kernel
    .globl kernel_end
kernel:
    SIMT_BLOCK_IDX  t0
    SIMT_BLOCK_DIM  t1
    SIMT_GRID_DIM   t2
    SIMT_THREAD_IDX t3
    SIMT_HW_TID     t4
    mul  t5, t0, t1          # blockIdx * blockDim
    add  t5, t5, t3          # gid
    slli t5, t5, 2
    add  a1, a0, t5          # a1 = &out[gid] (field 0)
    mul  t6, t1, t2          # total threads
    slli t6, t6, 2           # bytes per field
    sw   t0, 0(a1)
    add  a1, a1, t6
    sw   t1, 0(a1)
    add  a1, a1, t6
    sw   t2, 0(a1)
    add  a1, a1, t6
    sw   t3, 0(a1)
    add  a1, a1, t6
    sw   t4, 0(a1)
    ret
kernel_end:
