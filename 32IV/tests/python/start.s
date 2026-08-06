.section .text
.global _start

_start:
    la sp, _stack_top   # set up stack pointer before anything else runs
    call main            # jump into C main()

_end:
    j _end                # halt: infinite loop, nothing to return to
