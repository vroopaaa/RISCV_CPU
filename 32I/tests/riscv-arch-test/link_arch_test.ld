/* Used for BOTH the emulator-harness build and the QEMU_REFERENCE build --
 * they must share the same base address. qemu-riscv32 refuses to mmap a LOAD
 * segment at 0x0 (the reserved NULL page) with "Error mapping file: Operation
 * not permitted", so this uses a non-zero origin. Tests whose signature
 * encodes an absolute/PC-relative address (auipc, jal's return address, la)
 * would otherwise disagree between the two builds purely because of a
 * different link address, not any real CPU behavior difference -- this file
 * existing separately from tests/python/link.ld (origin 0x0, fine for every
 * other test suite here since none of them inspect absolute addresses) is
 * what avoids that class of false mismatch.
 */
MEMORY
{
  RAM (rwx) : ORIGIN = 0x00010000, LENGTH = 1M
}

SECTIONS
{
  .text : { *(.text*) } > RAM
  .data : { *(.data*) } > RAM
  .bss  : { *(.bss*) *(COMMON) } > RAM

  _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 4;
}
