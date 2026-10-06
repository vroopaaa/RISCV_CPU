	.file	"leaky_relu_32.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.data
	.align	2
	.type	x, @object
	.size	x, 128
x:
	.word	-16
	.word	-15
	.word	-14
	.word	-13
	.word	-12
	.word	-11
	.word	-10
	.word	-9
	.word	-8
	.word	-7
	.word	-6
	.word	-5
	.word	-4
	.word	-3
	.word	-2
	.word	-1
	.word	0
	.word	1
	.word	2
	.word	3
	.word	4
	.word	5
	.word	6
	.word	7
	.word	8
	.word	9
	.word	10
	.word	11
	.word	12
	.word	13
	.word	14
	.word	15
	.text
	.align	2
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-64
	sw	s0,60(sp)
	addi	s0,sp,64
 #APP
# 42 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 2, 0, a5, zero, zero
# 0 "" 2
 #NO_APP
	sw	a5,-48(s0)
	lw	a5,-48(s0)
	andi	a5,a5,255
	sw	a5,-24(s0)
	lui	a5,%hi(x)
	addi	a4,a5,%lo(x)
	lw	a5,-24(s0)
	slli	a5,a5,2
	add	a5,a4,a5
	lw	a5,0(a5)
	sw	a5,-28(s0)
	sw	zero,-20(s0)
	lw	a5,-28(s0)
	srli	a5,a5,31
	andi	a5,a5,0xff
	sw	a5,-32(s0)
	lw	a5,-32(s0)
	sw	a5,-44(s0)
	lw	a5,-44(s0)
	li	a4,0
 #APP
# 58 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 4, 0, zero, a5, a4
# 0 "" 2
 #NO_APP
	nop
	lw	a5,-32(s0)
	beq	a5,zero,.L3
	lw	a5,-28(s0)
	srai	a4,a5,31
	andi	a4,a4,7
	add	a5,a4,a5
	srai	a5,a5,3
	sw	a5,-20(s0)
.L3:
 #APP
# 63 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 5, 0, zero, zero, zero
# 0 "" 2
 #NO_APP
	nop
	lw	a5,-32(s0)
	seqz	a5,a5
	andi	a5,a5,0xff
	sw	a5,-36(s0)
	lw	a5,-36(s0)
	sw	a5,-52(s0)
	lw	a5,-52(s0)
	li	a4,0
 #APP
# 58 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 4, 0, zero, a5, a4
# 0 "" 2
 #NO_APP
	nop
	lw	a5,-36(s0)
	beq	a5,zero,.L4
	lw	a5,-28(s0)
	sw	a5,-20(s0)
.L4:
 #APP
# 63 "/home/roopi/Desktop/rsvp/32I/tests/simt/c_tests/../simt_isa.h" 1
	.insn r 0x2B, 5, 0, zero, zero, zero
# 0 "" 2
 #NO_APP
	nop
	li	a5,1048576
	sw	a5,-40(s0)
	lw	a5,-24(s0)
	slli	a5,a5,2
	lw	a4,-40(s0)
	add	a5,a4,a5
	lw	a4,-20(s0)
	sw	a4,0(a5)
	li	a5,0
	mv	a0,a5
	lw	s0,60(sp)
	addi	sp,sp,64
	jr	ra
	.size	main, .-main
	.ident	"GCC: (13.2.0-11ubuntu1+12) 13.2.0"
