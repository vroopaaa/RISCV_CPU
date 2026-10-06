	.file	"scalar_multiply.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.data
	.align	2
	.type	vec, @object
	.size	vec, 16
vec:
	.word	2
	.word	3
	.word	5
	.word	7
	.text
	.align	2
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sw	s0,28(sp)
	addi	s0,sp,32
	li	a5,4
	sw	a5,-32(s0)
	lw	a5,-32(s0)
 #APP
# 28 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 0, 0, zero, a5, zero
# 0 "" 2
 #NO_APP
	nop
 #APP
# 42 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 2, 0, a5, zero, zero
# 0 "" 2
 #NO_APP
	sw	a5,-28(s0)
	lw	a5,-28(s0)
	andi	a5,a5,255
	sw	a5,-20(s0)
	li	a5,1048576
	sw	a5,-24(s0)
	lui	a5,%hi(vec)
	addi	a4,a5,%lo(vec)
	lw	a5,-20(s0)
	slli	a5,a5,2
	add	a5,a4,a5
	lw	a4,0(a5)
	lw	a5,-20(s0)
	slli	a5,a5,2
	lw	a3,-24(s0)
	add	a5,a3,a5
	slli	a4,a4,2
	sw	a4,0(a5)
	li	a5,0
	mv	a0,a5
	lw	s0,28(sp)
	addi	sp,sp,32
	jr	ra
	.size	main, .-main
	.ident	"GCC: (13.2.0-11ubuntu1+12) 13.2.0"
