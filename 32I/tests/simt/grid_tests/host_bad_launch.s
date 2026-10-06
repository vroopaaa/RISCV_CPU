# host_bad_launch -- link with identity.s. Three launches that must all be
# skipped (host continues, no device output): threads_per_block 129,
# num_blocks 0, and the reserved funct3 1 of the LAUNCH opcode.
    .include "simt_macros.inc"
    .equ MARKER_ADDR, 0x180000
    .text
    .globl _start
    .globl _end
_start:
    la   sp, _stack_top
    li   a0, 0x100000
    la   t0, kernel
    li   t1, (1 << 16) | 129       # too many threads per block
    SIMT_LAUNCH t0, t1
    li   t1, (0 << 16) | 32        # no blocks
    SIMT_LAUNCH t0, t1
    li   t1, (1 << 16) | 32        # valid dims, but funct3 1 is reserved
    .insn r 0x5B, 1, 0, zero, t0, t1
    li   t2, 0x600D
    li   t3, MARKER_ADDR
    sw   t2, 0(t3)
_end:
    j    _end
