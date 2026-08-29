MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 4M
}

SECTIONS
{
  .text : { *(.text*) } > RAM
  .data : { *(.data*) } > RAM
  .bss  : { *(.bss*) *(COMMON) } > RAM

  _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 4;
}
