# vec_add -- c[i] = a[i] + b[i] for i < n, guarded by SPLIT/JOIN.
#   a0 = &args, args = { a_ptr, b_ptr, c_ptr, n } (run_grid_tests.py generates
#   the data file). Threads with gid >= n must leave c[gid] untouched.
    .include "simt_macros.inc"
    .text
    .globl kernel
kernel:
    lw   t0, 12(a0)          # n
    SIMT_BLOCK_IDX  t1
    SIMT_BLOCK_DIM  t2
    SIMT_THREAD_IDX t3
    mul  t4, t1, t2
    add  t4, t4, t3          # gid
    sltu t5, t4, t0          # gid < n
    SIMT_SPLIT t5, zero      # only in-range lanes run until JOIN
    slli t4, t4, 2
    lw   t1, 0(a0)
    add  t1, t1, t4
    lw   t1, 0(t1)           # a[gid]
    lw   t2, 4(a0)
    add  t2, t2, t4
    lw   t2, 0(t2)           # b[gid]
    add  t1, t1, t2
    lw   t3, 8(a0)
    add  t3, t3, t4
    sw   t1, 0(t3)           # c[gid]
    SIMT_JOIN
    ret
