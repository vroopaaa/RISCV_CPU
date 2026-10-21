/* SparseViT programs: 64 MB RAM, matching 32I/tests/basic/main.cpp's Memory.
   The top of RAM belongs to the stacks (docs/design/00-overview.md section 6):
   GPU lane stacks 0x3F70000-0x3FEFFFF, host stack 0x3FF0000-0x3FFFFFF. */
MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 64M
}

SECTIONS
{
  .text   : { *(.text*) } > RAM
  .rodata : { *(.rodata*) *(.srodata*) } > RAM
  .data   : { *(.data*) *(.sdata*) } > RAM
  .bss    : { *(.sbss*) *(.bss*) *(COMMON) } > RAM
  _program_end = .;

  _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 4;

  ASSERT(_program_end <= 0x3F70000, "program overlaps the GPU lane stacks (ends above 0x3F70000)")
}
