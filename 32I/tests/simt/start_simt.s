.section .text
.global _start

# SIMT-specific entry point -- unlike tests/python/start.s (one shared sp
# for the single scalar thread), every lane here executes this same
# instruction stream with its OWN registers but the SAME shared Memory*
# (see docs/CUDA/progress.md's "shared stack" finding: at -O0, even the
# simplest C function spills locals to the stack via sp/s0, and if every
# lane's sp pointed at the same address, those spills would silently
# clobber each other across lanes -- confirmed by a real bug: scalar_multiply
# only wrote lane 3's result because lanes 0-2's stack-spilled `idx` got
# overwritten by later lanes before being read back).
#
# Fix: give each lane its own private stack slice, offset from the top of
# RAM by its own thread id * STACK_SIZE_PER_THREAD, computed from the real
# TID instruction before main() (or anything that touches the stack) runs.
_start:
    .insn r 0x2B, 2, 0, t0, zero, zero   # t0 = simt_tid() -- packed id
    andi  t0, t0, 0xFF                    # t0 = thread_id (low byte)
    addi  t0, t0, 1                        # +1 so thread 0's slice doesn't start
                                             # exactly at _stack_top (leaves it
                                             # fully inside RAM with the sub below)
    li    t1, 0x400                          # STACK_SIZE_PER_THREAD = 1KB/lane,
                                               # plenty for -O0 locals in these kernels
    mul   t0, t0, t1                          # t0 = (thread_id+1) * 0x400
    la    sp, _stack_top
    sub   sp, sp, t0                           # this lane's own stack top
    call main

_end:
    j _end                # halt: infinite loop, nothing to return to
