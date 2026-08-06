.section .text
.global _start

_start:
    la sp, _stack_top   # set up stack pointer before anything else runs
    call main            # jump into your C main()

_end:
    j _end                # infinite loop -- halts here since there's no OS to return to