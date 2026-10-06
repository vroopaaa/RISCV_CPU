# runaway -- never returns. The first block must hit the issue cap, and the
# launcher must abandon the rest of the grid instead of running every block
# into the cap.
    .include "simt_macros.inc"
    .text
    .globl kernel
kernel:
1:  addi t0, t0, 1
    j    1b
