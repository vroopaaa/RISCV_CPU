# system_other -- SYSTEM instructions other than the two counter reads are
# not supported: each prints an error and is ignored (no register written).
    .text
    .globl _start
    .globl _end
_start:
    li   a0, 0x1234
    li   a1, 0x5678
    li   a2, 0x9abc
    li   t0, 7
bad_csrrw:
    csrrw a0, cycle, t0            # write to a read-only counter
bad_csrrs:
    csrrs a1, cycle, t0            # rs1 != x0: also a write
bad_instret:
    csrr  a2, instret              # a counter that isn't implemented
bad_ecall:
    ecall
bad_ebreak:
    ebreak
_end:
    j    _end
