// Custom-1 (opcode 0x2B) SIMT control instructions, wrapped as C macros over
// `.insn` inline asm -- same convention tests/npu/c_tests/*.c already use
// for the NPU's custom-0 opcode (0x0B). No compiler/assembler changes:
// ordinary riscv64-unknown-elf-gcc compiles these as plain R-type words.
// See docs/CUDA/plan.md for the funct3 table these match.
#ifndef SIMT_ISA_H
#define SIMT_ISA_H
#include <stdint.h>

// __attribute__((always_inline)) on every one of these is NOT a perf
// tweak -- it's required for correctness. `static inline` alone is just a
// hint GCC ignores at -O0, so these would otherwise compile to real
// jal/ret function calls. That's fatal specifically for simt_split/
// simt_join: SPLIT narrows the active mask *while some lanes are still
// inside that call frame* (sp/s0/ra pointing at simt_split's frame, not
// the caller's); when JOIN later reactivates those lanes, the shared warp
// pc has moved on to wherever the surviving lanes got to -- often a
// completely different call-nesting depth -- so the reactivated lanes read
// memory relative to a stale frame pointer that no longer matches what
// that pc expects (manifests as bogus addresses like 0xffffffe4, i.e. a
// frame pointer that's effectively gone to ~0). Forcing real inlining
// keeps the whole SPLIT..JOIN region as straight-line code in the
// *caller's* own frame, so there's no nested frame for a masked-off lane
// to get stuck inside. See docs/CUDA/progress.md for how this was found.

// rs1 = count -> tmask = (1 << count) - 1 for the calling warp.
static inline __attribute__((always_inline)) void simt_tmc(uint32_t count) {
    asm volatile(".insn r 0x2B, 0, 0, zero, %0, zero" :: "r"(count));
}

// rs1 = count, rs2 = addr -> activate `count` warps at pc = addr.
// Not implemented on the C++ side yet (needs the Phase 4 scheduler) --
// included here for forward compatibility, unused by today's kernels.
static inline __attribute__((always_inline)) void simt_wspawn(uint32_t count, uint32_t addr) {
    asm volatile(".insn r 0x2B, 1, 0, zero, %0, %1" :: "r"(count), "r"(addr));
}

// rd = pack_tid(warp, thread) for the calling thread -- see
// SIMTCore::pack_tid: [23:16] flat index | [15:8] warp_id | [7:0] thread_id.
static inline __attribute__((always_inline)) uint32_t simt_tid(void) {
    uint32_t id;
    asm volatile(".insn r 0x2B, 2, 0, %0, zero, zero" : "=r"(id));
    return id;
}

// rs1 = barrier id, rs2 = count. Not implemented on the C++ side yet
// (needs Phase 4) -- included for forward compatibility.
static inline __attribute__((always_inline)) void simt_bar(uint32_t id, uint32_t count) {
    asm volatile(".insn r 0x2B, 3, 0, zero, %0, %1" :: "r"(id), "r"(count));
}

// rs1 = per-lane predicate (0/1): push (old_mask, reconv_pc) onto the
// IPDOM stack, narrow tmask to lanes where predicate != 0. reconv_pc (rs2)
// is carried in the stack entry but not functionally used by JOIN yet (see
// docs/CUDA/plan.md), so this passes a fixed 0 rather than computing a
// real return address.
static inline __attribute__((always_inline)) void simt_split(uint32_t pred) {
    asm volatile(".insn r 0x2B, 4, 0, zero, %0, %1" :: "r"(pred), "r"(0u));
}

// Pop the IPDOM stack, restore tmask. No operands.
static inline __attribute__((always_inline)) void simt_join(void) {
    asm volatile(".insn r 0x2B, 5, 0, zero, zero, zero");
}

// rs1 = per-lane predicate: AND tmask with it, no stack push (caller
// restores manually, e.g. via another simt_pred() or simt_tmc()).
static inline __attribute__((always_inline)) void simt_pred(uint32_t pred) {
    asm volatile(".insn r 0x2B, 6, 0, zero, %0, zero" :: "r"(pred));
}

#endif // SIMT_ISA_H
