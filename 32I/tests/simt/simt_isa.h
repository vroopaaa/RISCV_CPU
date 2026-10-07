// Custom-1 (opcode 0x2B) SIMT control instructions, wrapped as C macros over
// `.insn` inline asm -- same convention tests/npu/c_tests/*.c already use
// for the NPU's custom-0 opcode (0x0B). No compiler/assembler changes:
// ordinary riscv64-unknown-elf-gcc compiles these as plain R-type words.
// See docs/CUDA/plan.md for the funct3 table these match.
#ifndef SIMT_ISA_H
#define SIMT_ISA_H
#include <stdint.h>

// SIMT programs are compiled at -O0 only, for now. gcc's optimiser assumes the
// lanes that fail a simt_split() condition run their own (empty) path, and may
// move code onto it -- e.g. computing a default value only there. On this
// hardware those lanes are switched off until simt_join() and run nothing, so
// that value is never computed (found in review: 22/64 lanes right at -O2).
// The real fix is a compiler that places SPLIT/JOIN itself.
#ifdef __OPTIMIZE__
#error "SIMT programs must be compiled at -O0 for now (see the note in simt_isa.h)"
#endif

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
    asm volatile(".insn r 0x2B, 0, 0, zero, %0, zero" :: "r"(count) : "memory");
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
//
// Returns `pred` -- branch on the RETURNED value, not on the original:
//     if (simt_split(cond)) { ... }
//     simt_join();
// The "+r" makes the value opaque to the compiler. Without it, gcc -O2 sees
// that two regions test the same condition (e.g. the if/else halves of a
// divergent branch) and compiles them as ONE branch placed after the first
// SPLIT, with copies of the SPLIT/JOIN on each side -- the whole warp then
// follows whichever side the first region's lanes took, and the other lanes
// silently skip their side (found by tests/simt/grid_tests/c_relu.c at -O2).
// "memory": loads/stores inside the region must stay inside it.
static inline __attribute__((always_inline)) uint32_t simt_split(uint32_t pred) {
    asm volatile(".insn r 0x2B, 4, 0, zero, %0, %1" : "+r"(pred) : "r"(0u) : "memory");
    return pred;
}

// Pop the IPDOM stack, restore tmask. No operands.
static inline __attribute__((always_inline)) void simt_join(void) {
    asm volatile(".insn r 0x2B, 5, 0, zero, zero, zero" ::: "memory");
}

// rs1 = per-lane predicate: AND tmask with it, no stack push (caller
// restores manually, e.g. via another simt_pred() or simt_tmc()).
static inline __attribute__((always_inline)) void simt_pred(uint32_t pred) {
    asm volatile(".insn r 0x2B, 6, 0, zero, %0, zero" :: "r"(pred) : "memory");
}

// ---------------------------------------------------------------------------
// Grid launch (docs/CUDA/grid_launch_plan.md 4.3) -- the CUDA-style layer.
// A kernel is a plain C function `void kernel(void* args)`; the launcher gives
// every thread its own stack, sets a0 = args, and ends the thread when the
// function returns.
// ---------------------------------------------------------------------------

// Identity values (custom-1 funct3 7, funct7 selects which -- SIMTCore::IdentOp).
static inline __attribute__((always_inline)) uint32_t simt_block_idx(void) {   // blockIdx.x
    uint32_t v;
    asm volatile(".insn r 0x2B, 7, 0, %0, zero, zero" : "=r"(v));
    return v;
}
static inline __attribute__((always_inline)) uint32_t simt_block_dim(void) {   // blockDim.x
    uint32_t v;
    asm volatile(".insn r 0x2B, 7, 1, %0, zero, zero" : "=r"(v));
    return v;
}
static inline __attribute__((always_inline)) uint32_t simt_grid_dim(void) {    // gridDim.x
    uint32_t v;
    asm volatile(".insn r 0x2B, 7, 2, %0, zero, zero" : "=r"(v));
    return v;
}
// Unique physical thread id across all SMs: (sm*WARPS_RESIDENT + warp_slot)*32 + lane.
static inline __attribute__((always_inline)) uint32_t simt_hw_tid(void) {
    uint32_t v;
    asm volatile(".insn r 0x2B, 7, 3, %0, zero, zero" : "=r"(v));
    return v;
}
// threadIdx.x: bits [23:16] of the packed TID.
static inline __attribute__((always_inline)) uint32_t simt_thread_idx(void) {
    return (simt_tid() >> 16) & 0xFF;
}
// blockIdx.x * blockDim.x + threadIdx.x
static inline __attribute__((always_inline)) uint32_t simt_global_id(void) {
    return simt_block_idx() * simt_block_dim() + simt_thread_idx();
}

// Host only: run `kernel` over num_blocks x threads_per_block threads and
// return once every block is done -- kernel<<<num_blocks, threads_per_block>>>(args).
// LAUNCH (custom-2, 0x5B) reads the kernel argument implicitly from a0, so
// `args` is pinned there. The "memory" clobber is required: the GPU reads and
// writes memory behind the compiler's back, so stores before the launch must
// really happen first, and nothing loaded before it may be reused after it.
static inline __attribute__((always_inline)) void simt_launch(void (*kernel)(void*), uint32_t threads_per_block,
                                                              uint32_t num_blocks, void* args) {
    register void* a0 asm("a0") = args;
    uint32_t dims = (num_blocks << 16) | threads_per_block;
    asm volatile(".insn r 0x5B, 0, 0, zero, %0, %1" :: "r"(kernel), "r"(dims), "r"(a0) : "memory");
}

#endif // SIMT_ISA_H
