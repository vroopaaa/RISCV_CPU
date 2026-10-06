"""
Kernels as data: plain lists of symbolic ops (see simt_model.py's op
dispatch table / README.md for the format). No RISC-V, no compiler --
these are hand-written the way a human would hand-assemble them, standing
in for what riscv64-unknown-elf-gcc would eventually produce from C.

Register convention used by the kernels below (purely a convention of this
file, not enforced by simt_model.py):
  r0 = idx (this lane's thread id, from TID)
  r1 = scratch / address register
  r2 = loaded value
  r3 = scalar constant
  r4 = result
"""

import math

# ---------------------------------------------------------------------
# Reduced version of the pasted CUDA example:
#   __global__ void scalarMultiplyKernel(float *vec, float scalar, int n)
#   vec[idx] = vec[idx] * scalar;
# with N = 4, one block, threadsPerBlock = 4 -- so idx == lane index and
# the `if (idx < n)` boundary check is always true (every lane owns
# exactly one element). No divergence, so this kernel is fair game before
# BRANCH/SPLIT/JOIN exist.
#
# vec[] lives in shared_mem[0:4]; the scalar is baked in via LOADI for now
# (a real version would pass it in a register/shared-mem slot set up by
# the "host" before launch, same as vec[] is).
# ---------------------------------------------------------------------

SCALAR_MULTIPLY_SCALAR_VALUE = 4  # keep it an int for this prototype (float* not needed yet)

SCALAR_MULTIPLY_KERNEL = [
    ("TID",),                               # pc 0: r0 = lane index -> this lane's idx
    ("LOADI", 3, SCALAR_MULTIPLY_SCALAR_VALUE),  # pc 1: r3 = scalar
    ("LOAD", 2, 0),                          # pc 2: r2 = shared_mem[r0] (r0 doubles as addr reg)
    ("MUL", 4, 2, 3),                        # pc 3: r4 = r2 * r3
    ("STORE", 0, 4),                         # pc 4: shared_mem[r0] = r4
    ("HALT",),                               # pc 5
]


# ---------------------------------------------------------------------
# Add a second, even smaller kernel here once the above passes -- e.g. a
# masked variant (tmask != 0b1111) to exercise per-lane masking, or a
# BRANCH-using kernel to exercise lockstep branch semantics.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Manufactured divergent branch -- the one hand-written test SPLIT/JOIN
# need to prove the IPDOM stack mechanism itself (see simt_model.py's
# Warp.ipdom_stack / _op_split / _op_join docstrings). Not meant to look
# like compiler output -- this is "if (idx < 2) result = 100; else result
# = 200; shared_mem[idx] = result;" with idx = lane index, so lanes 0,1
# take the if-branch and lanes 2,3 take the else-branch: genuine
# per-lane divergence on a single SPLIT.
#
# Register convention (this kernel only):
#   r0 = idx (from TID)
#   r5 = predicate (idx < 2), from SLT
#   r6 = result
#
# Expected shared_mem after running: [100, 100, 200, 200]
# ---------------------------------------------------------------------

DIVERGENT_BRANCH_KERNEL = [
    ("TID",),            # pc 0:  r0 = idx
    ("LOADI", 1, 2),     # pc 1:  r1 = 2 (comparison threshold n)
    ("SLT", 5, 0, 1),    # pc 2:  r5 = (idx < 2) ? 1 : 0
    ("SPLIT", 5, 6),     # pc 3:  narrow to lanes where r5 != 0 (0,1); reconv_pc=6
    ("LOADI", 6, 100),   # pc 4:  if-body, only lanes 0,1 active
    ("JOIN",),           # pc 5:  restore mask to all 4 lanes
    # else condition = NOT(idx < 2) = idx >= 2, i.e. the complement of r5
    # above -- r5 is untouched by SPLIT/JOIN (they only ever touch
    # warp.tmask/ipdom_stack, never registers), so it still holds every
    # lane's original (idx < 2) result here. Negating it (instead of a
    # second, easy-to-get-the-boundary-wrong comparison like "2 < idx",
    # which wrongly excludes idx==2) avoids re-deriving the same condition
    # a second time: r5_not = 1 - r5, via existing ops (no NOT op exists).
    ("LOADI", 7, -1),    # pc 6:  r7 = -1
    ("MUL", 5, 5, 7),    # pc 7:  r5 = -r5
    ("ADDI", 5, 5, 1),   # pc 8:  r5 = 1 - (original r5) = NOT(idx < 2), i.e. idx >= 2
    ("SPLIT", 5, 12),    # pc 9:  narrow to lanes where r5 != 0 (2,3)
    ("LOADI", 6, 200),   # pc 10: else-body, only lanes 2,3 active
    ("JOIN",),           # pc 11: restore mask to all 4 lanes
    ("STORE", 0, 6),     # pc 12: shared_mem[idx] = r6, all 4 lanes
    ("HALT",),           # pc 13
]


# ---------------------------------------------------------------------
# Softmax over a T=4 vector, using a fixed-point exp() lookup table instead
# of real floating point (nothing in this ISA has floats or an exp op).
#
# Fixed-point convention: every table/intermediate value is exp(x)*SCALE,
# rounded to the nearest int. Final output is per-mille (i.e. *1000, summing
# to ~1000 across all 4 lanes, modulo integer-division rounding loss).
#
# Memory layout (shared_mem), chosen so the kernel can address everything
# via a single ADDI + LOAD/STORE per access:
#   [0:4)    input vector x[0..3], x[i] must be an int in LUT_DOMAIN
#   [4:9)    LUT_BASE: exp(x)*SCALE for x in LUT_DOMAIN, indexed by
#            (x - LUT_MIN) -- i.e. table_addr = x + (LUT_BASE - LUT_MIN)
#   [9:13)   EXP_BASE: lane i's own exp(x[i])*SCALE, written by every lane
#            so every OTHER lane can read it back for the sum (shared_mem
#            is this model's only cross-lane communication channel --
#            registers are strictly private per lane)
#   [13:17)  OUT_BASE: final per-mille result, kept separate from the input
#            so a test can diff both at once
#
# No BAR/barrier instruction exists yet (deferred along with the rest of
# Phase 4's scheduler work) -- this kernel doesn't need one anyway, since a
# single warp is lockstep by construction: every active lane finishes
# instruction N before any lane starts instruction N+1, so by the time the
# "sum" block runs, every lane's STORE into EXP_BASE has already happened.
# A multi-warp version of this would need a real barrier.
# ---------------------------------------------------------------------

SOFTMAX_LUT_MIN = -2
SOFTMAX_LUT_MAX = 2
SOFTMAX_LUT_DOMAIN = range(SOFTMAX_LUT_MIN, SOFTMAX_LUT_MAX + 1)  # -2..2 inclusive
SOFTMAX_SCALE = 1000       # fixed-point scale for exp() table entries
SOFTMAX_OUT_SCALE = 1000   # output is per-mille

SOFTMAX_VEC_LEN = 4  # == THREADS_PER_WARP, one vector element per lane (not imported to avoid a simt_model <-> kernels import cycle)

SOFTMAX_LUT_BASE = 4
SOFTMAX_EXP_BASE = SOFTMAX_LUT_BASE + len(SOFTMAX_LUT_DOMAIN)   # 9
SOFTMAX_OUT_BASE = SOFTMAX_EXP_BASE + SOFTMAX_VEC_LEN           # 13
SOFTMAX_SHARED_MEM_SIZE = SOFTMAX_OUT_BASE + SOFTMAX_VEC_LEN    # 17

# table_addr = x + SOFTMAX_TABLE_OFFSET maps x in SOFTMAX_LUT_DOMAIN directly
# onto [SOFTMAX_LUT_BASE, SOFTMAX_LUT_BASE + len(domain)).
SOFTMAX_TABLE_OFFSET = SOFTMAX_LUT_BASE - SOFTMAX_LUT_MIN  # 6

SOFTMAX_LUT = [round(math.exp(x) * SOFTMAX_SCALE) for x in SOFTMAX_LUT_DOMAIN]

LEAKY_RELU_OUT_BASE = 4  # x lives in shared_mem[0:4]; result in shared_mem[4:8]

# ---------------------------------------------------------------------
# Leaky ReLU -- a real activation function, and a more realistic divergence
# example than DIVERGENT_BRANCH_KERNEL above: that one just LOADI'd a
# different constant per branch, which doesn't show why divergence actually
# costs anything. Here both branches do genuinely different arithmetic on
# the lane's OWN data:
#   y[idx] = x[idx]        if x[idx] >= 0
#   y[idx] = x[idx] / 8    if x[idx] <  0   (integer stand-in for the usual
#                                             "leaky slope" of ~0.1-0.3;
#                                             truncating division, same as
#                                             real hardware's DIV)
#
# Same two-SPLIT/JOIN-blocks-in-program-order pattern as
# DIVERGENT_BRANCH_KERNEL, but this time both branches write into the SAME
# result register (r4) instead of each doing its own STORE, with a single
# STORE after the second JOIN -- the more common shape for compiled
# if/else-with-a-merge-point code.
#
# reconv_pc convention (both SPLITs below): the pc of the matching JOIN's
# own pc + 1, i.e. the first instruction that runs once lanes reconverge.
#
# Register convention (this kernel only):
#   r0 = idx          r1 = x[idx]              r2 = 0 (comparison constant)
#   r3 = predicate     r4 = result (shared)      r5 = divisor (8)
#   r6 = -1 (for negating r3)                    r7 = output address
# ---------------------------------------------------------------------

LEAKY_RELU_KERNEL = [
    ("TID",),            # pc 0:  r0 = idx
    ("LOAD", 1, 0),      # pc 1:  r1 = x[idx] = shared_mem[idx]
    ("LOADI", 2, 0),     # pc 2:  r2 = 0
    ("SLT", 3, 1, 2),    # pc 3:  r3 = (x[idx] < 0) ? 1 : 0
    ("SPLIT", 3, 8),     # pc 4:  narrow to negative lanes; reconv_pc = 8 (JOIN@7 + 1)
    ("LOADI", 5, 8),     # pc 5:  r5 = 8 (the "leaky" divisor), negative lanes only
    ("DIV", 4, 1, 5),    # pc 6:  r4 = x[idx] / 8 (truncating), negative lanes only
    ("JOIN",),           # pc 7:  restore mask to all 4 lanes
    # else condition = NOT(x[idx] < 0) = x[idx] >= 0 -- negate r3 the same
    # way DIVERGENT_BRANCH_KERNEL does (no NOT op exists): r3 untouched by
    # SPLIT/JOIN, so it still holds every lane's original predicate here.
    ("LOADI", 6, -1),    # pc 8:  r6 = -1
    ("MUL", 3, 3, 6),    # pc 9:  r3 = -r3
    ("ADDI", 3, 3, 1),   # pc 10: r3 = 1 - (original r3) = NOT(x<0), i.e. x>=0
    ("SPLIT", 3, 14),    # pc 11: narrow to non-negative lanes; reconv_pc = 14 (JOIN@13 + 1)
    ("ADD", 4, 1, 2),    # pc 12: r4 = x[idx] + 0 = x[idx] (pass through), non-negative lanes only
    ("JOIN",),           # pc 13: restore mask to all 4 lanes
    ("ADDI", 7, 0, LEAKY_RELU_OUT_BASE),  # pc 14: r7 = OUT_BASE + idx
    ("STORE", 7, 4),     # pc 15: shared_mem[OUT_BASE + idx] = r4, all 4 lanes
    ("HALT",),           # pc 16
]


SOFTMAX_KERNEL = [
    # ---- exp(x[idx]) via the lookup table ----
    ("TID",),                                  # pc 0:  r0 = idx
    ("LOAD", 1, 0),                            # pc 1:  r1 = shared_mem[idx] = x[idx]
    ("ADDI", 2, 1, SOFTMAX_TABLE_OFFSET),      # pc 2:  r2 = table address for x[idx]
    ("LOAD", 3, 2),                            # pc 3:  r3 = LUT[x[idx]] = exp(x[idx]) * SCALE
    ("ADDI", 4, 0, SOFTMAX_EXP_BASE),          # pc 4:  r4 = EXP_BASE + idx
    ("STORE", 4, 3),                           # pc 5:  shared_mem[EXP_BASE + idx] = r3
    # ---- sum of all 4 lanes' exp values (every lane computes the same
    #      total independently -- no reduction primitive exists yet) ----
    ("LOADI", 4, SOFTMAX_EXP_BASE + 0),        # pc 6
    ("LOAD", 5, 4),                            # pc 7:  r5 = exp[0]
    ("LOADI", 4, SOFTMAX_EXP_BASE + 1),        # pc 8
    ("LOAD", 6, 4),                            # pc 9:  r6 = exp[1]
    ("ADD", 5, 5, 6),                          # pc 10: r5 = exp[0] + exp[1]
    ("LOADI", 4, SOFTMAX_EXP_BASE + 2),        # pc 11
    ("LOAD", 6, 4),                            # pc 12: r6 = exp[2]
    ("ADD", 5, 5, 6),                          # pc 13: r5 += exp[2]
    ("LOADI", 4, SOFTMAX_EXP_BASE + 3),        # pc 14
    ("LOAD", 6, 4),                            # pc 15: r6 = exp[3]
    ("ADD", 5, 5, 6),                          # pc 16: r5 = sum of all 4 exp values
    # ---- result = exp(x[idx]) * OUT_SCALE / sum ----
    ("LOADI", 6, SOFTMAX_OUT_SCALE),           # pc 17: r6 = 1000
    ("MUL", 3, 3, 6),                          # pc 18: r3 = exp(x[idx]) * SCALE * 1000
    ("DIV", 3, 3, 5),                          # pc 19: r3 = r3 / sum -- per-mille probability
    ("ADDI", 7, 0, SOFTMAX_OUT_BASE),          # pc 20: r7 = OUT_BASE + idx
    ("STORE", 7, 3),                           # pc 21: shared_mem[OUT_BASE + idx] = result
    ("HALT",),                                 # pc 22
]

