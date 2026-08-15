	.file	"main.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	2
	.type	custom_mul, @function
custom_mul:
	addi	sp,sp,-48
	sw	s0,44(sp)
	addi	s0,sp,48
	sw	a0,-36(s0)
	sw	a1,-40(s0)
	lw	a5,-36(s0)
	lw	a4,-40(s0)
 #APP
# 5 "/home/roopi/Desktop/rsvp/32I/tests/inline_assembly/main.c" 1
	.insn r 0x0B, 0, 0, a5, a5, a4
# 0 "" 2
 #NO_APP
	sw	a5,-20(s0)
	lw	a5,-20(s0)
	mv	a0,a5
	lw	s0,44(sp)
	addi	sp,sp,48
	jr	ra
	.size	custom_mul, .-custom_mul
	.align	2
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sw	ra,28(sp)
	sw	s0,24(sp)
	addi	s0,sp,32
	li	a5,5
	sw	a5,-20(s0)
	li	a5,10
	sw	a5,-24(s0)
	lw	a5,-20(s0)
	lw	a4,-24(s0)
	mv	a1,a4
	mv	a0,a5
	call	custom_mul
	mv	a5,a0
	sw	a5,-28(s0)
	li	a5,0
	mv	a0,a5
	lw	ra,28(sp)
	lw	s0,24(sp)
	addi	sp,sp,32
	jr	ra
	.size	main, .-main
	.ident	"GCC: (13.2.0-11ubuntu1+12) 13.2.0"
