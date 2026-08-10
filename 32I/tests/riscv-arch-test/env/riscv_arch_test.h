// riscv_arch_test.h
//
// Minimal, hand-written stand-in for the official riscv-arch-test ACT4
// framework header of the same name. The real one pulls in a full M-mode
// boot sequence (mtvec/mstatus/mepc setup, trap handlers, PMP, vector,
// hypervisor support) driven by a UDB config + Sail-computed self-checking
// signatures -- none of which this toy emulator implements or needs for an
// unprivileged, non-trapping instruction test like M-div-00.S (it only ever
// executes `div`/`mv` plus these macros; grep confirms no CSR instructions).
//
// Instead of self-checking (comparing against a Sail-computed reference
// baked into the binary), this just records every result word to a plain
// memory region between begin_signature/end_signature. run_arch_test.py
// dumps that region after the run and independently computes the expected
// DIV results in Python to diff against.
//
// Implements only the macros actually used by the test files under
// assembly_files/ (M-extension + base-I): RVTEST_BEGIN, RVTEST_CODE_END,
// RVTEST_DATA_BEGIN, RVTEST_DATA_END, RVTEST_SIG_SETUP,
// RVTEST_TESTDATA_LOAD_INT, RVTEST_SIGUPD, plus the RV32-only aliases the I
// suite's test bodies reference directly (not just inside our own macros):
// LA, LI, SREG, LREG, REGWIDTH. These mirror tests/env/utils.h's RV32
// (UDB_MXLEN==32) definitions, minus the alignment-determinism bookkeeping
// (.p2align/.option rvc dance) the official LA/LI wrap them in -- that's for
// keeping code size identical across DUT configs with different compressed-
// instruction support, which doesn't matter for a single fixed toolchain.

#ifndef _RISCV_ARCH_TEST_H
#define _RISCV_ARCH_TEST_H

#define SIG_STRIDE 4
#define REGWIDTH 4
#define SREG sw
#define LREG lw
#define LA(reg, val) la reg, val
#define LI(reg, imm) li reg, imm

// Sets up the entry point, stack, and the two pointer registers the test
// body uses throughout: x2 (signature/result pointer) and x3 (test data
// pointer, walking the pre-generated random operand table).
// .option norelax: without this, the linker can "simplify" the la (auipc+addi)
// sequences below into something that computes the wrong address -- exactly
// the bug the official framework's own RVTEST_BEGIN guards against.
#define RVTEST_BEGIN                    \
  .option push                         ;\
  .option norelax                      ;\
  .section .text.init                  ;\
  .global _start                       ;\
  _start:                              ;\
    la sp, _stack_top                  ;\
    la x2, begin_signature             ;\
    la x3, rvtest_data_begin           ;\
  .option pop

// Halts immediately after the last real test instruction, still in .text --
// critical to do it *here*, not later. Without an explicit halt right after
// the test body, the CPU's PC just falls through into whatever comes next in
// memory, which is the .data section (the random operand table below), and
// starts executing that as if it were code.
//
// Two variants, chosen by whether QEMU_REFERENCE is defined at compile time:
//   - normal build (this emulator): jump to itself. This codebase's halt
//     idiom -- CPU::execute() sets halted when next_pc == pc, same as
//     start.s's `_end: j _end`.
//   - QEMU_REFERENCE build: qemu-riscv32 runs this as a real Linux process
//     via user-mode emulation, so instead of spinning forever it writes the
//     whole signature region to stdout with a `write` syscall and exits with
//     `exit` -- giving run_arch_test.py a real, independently-implemented
//     RISC-V reference (QEMU) to diff against instead of hand-computing
//     expected DIV results in Python.
#ifdef QEMU_REFERENCE
#define RVTEST_CODE_END                    \
  .align 2                                ;\
    la a1, begin_signature                ;\
    la a2, end_signature                  ;\
    sub a2, a2, a1                        ;\
    li a0, 1        /* fd = stdout */     ;\
    li a7, 64       /* sys_write */       ;\
    ecall                                 ;\
    li a0, 0                              ;\
    li a7, 93       /* sys_exit */        ;\
    ecall
#else
#define RVTEST_CODE_END                \
  .align 2                             ;\
  _arch_test_halt:                     ;\
    j _arch_test_halt
#endif

// scratch: a 264-byte distinctively-patterned buffer that load/store test
// cases use as a base address for testing various byte/halfword/word offsets
// -- not part of the signature region itself. Copied verbatim (same fill
// pattern, same 256-byte alignment) from the official framework's
// rvtest_setup.h so any test that reads back an unmodified scratch byte gets
// the identical value in both this build and the QEMU_REFERENCE build.
#define RVTEST_DATA_BEGIN                                    \
  .section .data                                            ;\
  .p2align 8                                                ;\
  .global scratch                                           ;\
  scratch:                                                  ;\
    .dword 0xDEAD0001FFFEBEEF, 0xDEAD0002FFFDBEEF           ;\
    .dword 0xDEAD0003FFFCBEEF, 0xDEAD0004FFFBBEEF           ;\
    .dword 0xDEAD0005FFFABEEF, 0xDEAD0006FFF9BEEF           ;\
    .dword 0xDEAD0007FFF8BEEF, 0xDEAD0008FFF7BEEF           ;\
    .dword 0xDEAD0009FFF6BEEF, 0xDEAD000AFFF5BEEF           ;\
    .dword 0xDEAD000BFFF4BEEF, 0xDEAD000CFFF3BEEF           ;\
    .dword 0xDEAD000DFFF2BEEF, 0xDEAD000EFFF1BEEF           ;\
    .dword 0xDEAD000FFFF0BEEF, 0xDEAD0010FFEFBEEF           ;\
    .dword 0xDEAD0011FFEEBEEF, 0xDEAD0012FFEDBEEF           ;\
    .dword 0xDEAD0013FFECBEEF, 0xDEAD0014FFEBBEEF           ;\
    .dword 0xDEAD0015FFEABEEF, 0xDEAD0016FFE9BEEF           ;\
    .dword 0xDEAD0017FFE8BEEF, 0xDEAD0018FFE7BEEF           ;\
    .dword 0xDEAD0019FFE6BEEF, 0xDEAD001AFFE5BEEF           ;\
    .dword 0xDEAD001BFFE4BEEF, 0xDEAD001CFFE3BEEF           ;\
    .dword 0xDEAD001DFFE2BEEF, 0xDEAD001EFFE1BEEF           ;\
    .dword 0xDEAD001FFFE0BEEF, 0xDEAD0020FFDFBEEF           ;\
    .dword 0xDEAD0021FFDEBEEF                                ;\
  .align 2                                                   ;\
  .global rvtest_data_begin                                 ;\
  rvtest_data_begin:

#define RVTEST_DATA_END                \
  .global rvtest_data_end              ;\
  rvtest_data_end:

// Reserves SIGUPD_COUNT (defined by the test file before this include)
// words of scratch memory for RVTEST_SIGUPD to fill in. Execution never
// reaches here -- RVTEST_CODE_END already halted -- this just defines the
// begin_signature/end_signature bounds run_arch_test.py reads via `nm`.
#define RVTEST_SIG_SETUP               \
  .align 2                             ;\
  .global begin_signature              ;\
  begin_signature:                     ;\
    .fill SIGUPD_COUNT, SIG_STRIDE, 0  ;\
  .global end_signature                ;\
  end_signature:

// Loads a word from the test data table into _DEST_REG and advances the
// pointer to the next word.
#define RVTEST_TESTDATA_LOAD_INT(_DATA_PTR, _DEST_REG)  \
  lw _DEST_REG, 0(_DATA_PTR)                            ;\
  addi _DATA_PTR, _DATA_PTR, SIG_STRIDE

// Records the result in _R to the signature region and advances the pointer.
// _LINK_REG/_TEMP_REG/_INST_PTR/_STR_PTR (used by the official self-checking
// version to jump to a failure handler with debug info) are unused here --
// mismatches are found externally by run_arch_test.py, not on-target.
#define RVTEST_SIGUPD(_SIG_PTR, _LINK_REG, _TEMP_REG, _R, _INST_PTR, _STR_PTR)  \
  sw _R, 0(_SIG_PTR)                                                           ;\
  addi _SIG_PTR, _SIG_PTR, SIG_STRIDE

#endif // _RISCV_ARCH_TEST_H
