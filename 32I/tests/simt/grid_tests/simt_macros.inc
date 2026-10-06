# Assembler macros for the SIMT custom-1 ops (opcode 0x2B) -- the assembly
# twin of tests/simt/simt_isa.h. Each one expands to exactly ONE instruction
# word, `.insn r 0x2B, funct3, funct7, rd, rs1, rs2`, so a kernel's objdump
# listing lines up 1:1 with what the SM fetches. See include/SIMT.h (SimtOp /
# IdentOp) for the semantics.

# funct3 2: rd = packed TID -- [23:16] threadIdx | [15:8] warp in block | [7:0] lane
.macro SIMT_TID rd
    .insn r 0x2B, 2, 0, \rd, zero, zero
.endm

# funct3 4: push (tmask, reconv_pc), narrow tmask to lanes where pred != 0.
# reconv_pc is carried but not used by JOIN yet, so pass any register.
.macro SIMT_SPLIT pred, reconv
    .insn r 0x2B, 4, 0, zero, \pred, \reconv
.endm

# funct3 5: pop the IPDOM stack, restore tmask.
.macro SIMT_JOIN
    .insn r 0x2B, 5, 0, zero, zero, zero
.endm

# funct3 7 (SIMT_IDENT), funct7 selects the value -- rd = ...
.macro SIMT_BLOCK_IDX rd        # blockIdx
    .insn r 0x2B, 7, 0, \rd, zero, zero
.endm
.macro SIMT_BLOCK_DIM rd        # threads per block
    .insn r 0x2B, 7, 1, \rd, zero, zero
.endm
.macro SIMT_GRID_DIM rd         # blocks in the grid
    .insn r 0x2B, 7, 2, \rd, zero, zero
.endm
.macro SIMT_HW_TID rd           # unique physical thread id: (sm*W + warp_slot)*T + lane
    .insn r 0x2B, 7, 3, \rd, zero, zero
.endm

# Not an instruction of its own: rd = threadIdx, from the packed TID (2 instructions + TID).
.macro SIMT_THREAD_IDX rd
    SIMT_TID \rd
    srli \rd, \rd, 16
    andi \rd, \rd, 0xFF
.endm
