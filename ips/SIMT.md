# SIMTCore IP Specification & Integration Datasheet

**Module:** `SIMTCore`  
**Architecture:** 32-bit RISC-V (RV32IM) SIMT Streaming Multiprocessor  
**Document Version:** 1.0  
**Target Audience:** System Architects, Controller Designers, Runtime Engineers  

---

## 1. Overview

`SIMTCore` is a synthesizable/simulatable Streaming Multiprocessor (SM) IP designed for data-parallel execution under a Single Instruction, Multiple Threads (SIMT) execution model. It executes standard RV32IM machine code across multiple lockstep thread lanes, augmented by a custom SIMT control instruction extension (opcode `0x2B`).

### Key Characteristics
- **Execution Model:** Lockstep SIMD execution within a warp (32 lanes); MIMD context isolation across resident warps.
- **ISA Compatibility:** Unmodified RV32I base integer instructions + M extension (multiplication/division) + Custom-1 SIMT control extension.
- **Storage:** Banked physical register file, flat shared memory interface, hardware Immediate Post-Dominator (IPDOM) reconvergence stack per warp.
- **Dispatch Contract:** Dispatches and executes complete thread blocks up to the maximum resident capacity.

---

## 2. Hardware Configuration Parameters

The core parameters are statically fixed at hardware generation / compile time. Any external controller or dispatcher must configure workloads within these boundaries:

| Parameter | Value | Definition / Constraint |
|---|---|---|
| `THREADS_PER_WARP` ($T$) | **32** | Physical lanes executed in lockstep per warp. |
| `WARPS_RESIDENT` ($W$) | **4** | Number of physical warp context slots per core. |
| `NUM_ARCH_REGS` | **32** | Standard architectural registers per lane ($x0 \dots x31$). |
| `PHYS_REGS` | **4096** | Total physical registers per core ($T \times W \times 32$). |
| `MAX_THREADS_PER_BLOCK` | **128** | **Strict block size limit** ($T \times W = 32 \times 4$). |
| `STACK_BYTES_PER_THREAD` | **0x400** (1 KB) | Hardware stack allocation dedicated per physical lane. |
| `IPDOM_DEPTH` | **8** | Maximum nested `SPLIT`/`JOIN` divergence stack depth. |
| `BLOCK_RETURN_PC` | **`0xFFFFFFF0`** | Execution termination sentinel address (outside RAM). |

> **Critical Constraint:** A single thread block assigned to this IP **must not exceed 128 threads** ($1 \le \text{block\_dim} \le 128$). Blocks requiring more than 128 threads must be partitioned into multiple smaller blocks by the upstream system dispatcher.

---

## 3. Register File Organization

The register file is organized into **32 independent lane banks**:
$$\text{Storage}[t][w][r - 1] \quad \text{for } t \in [0, 31], \; w \in [0, 3], \; r \in [1, 31]$$

- **Banked Access:** Lane $t$ strictly accesses Bank $t$. All 32 active lanes read/write operands in parallel within a single cycle without bank conflicts.
- **$x0$ Hardwiring:** Register $x0$ has no physical storage. Reads always return `0x00000000`, and writes are discarded.
- **Zero-Initialization:** Upon dispatching a block, the core deterministically zeroes registers $x1 \dots x31$ across all lanes in allocated warps to eliminate cross-block data leakage.

---

## 4. Instruction Set & Execution Semantics

### 4.1 Base Instruction Broadcast
Instructions fetched from `warp.pc` are decoded once and broadcast across all 32 lanes. Only lanes with their corresponding bit asserted in `warp.tmask` execute the ALU operation, load/store, or writeback.

### 4.2 Branch & Jump Semantics
- **Uniform Conditional Branches (RV32I `0x63`):**  
  Standard RISC-V branches (`BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`) are **SIMT-uniform-only**. Every active lane in the warp must evaluate to the same branch decision. If active lanes disagree, an execution assertion is raised.
- **Divergent Conditions:**  
  Thread-divergent conditions must **not** use opcode `0x63` directly. They must use predicated execution via `SIMT_SPLIT` / `SIMT_JOIN`.
- **Unconditional Jumps (`JAL`, `JALR`):**  
  `JAL` is inherently uniform. `JALR` calculates its target address exclusively from the warp leader lane's register value ($rs1$).

### 4.3 Custom-1 SIMT Control Extension (Opcode `0x2B`)
Encoded as RISC-V R-type: `.insn r 0x2B, funct3, funct7, rd, rs1, rs2`.

| `funct3` | Mnemonic | Operands | Functional Description |
|---|---|---|---|
| `0` | `SIMT_TMC` | $rs1$ = count | Sets active thread mask: `tmask = (1 << count) - 1`. |
| `2` | `SIMT_TID` | $rd$ = destination | Returns packed thread coordinates (see layout below). |
| `4` | `SIMT_SPLIT` | $rs1$ = predicate, $rs2$ = reconv PC | Pushes old mask and reconvergence PC to IPDOM stack; narrows `tmask` to lanes where $rs1 \neq 0$. |
| `5` | `SIMT_JOIN` | *(none)* | Pops IPDOM stack; restores previous execution mask. |
| `6` | `SIMT_PRED` | $rs1$ = predicate | Bitwise-ANDs `tmask` with per-lane predicate (soft narrowing without stack push). |
| `7` | `SIMT_IDENT` | $rd$ = dest, `funct7` = selector | Returns block, grid, or hardware identity value. |

#### TID Register Bit Packing (`SIMT_TID`)
- `[7:0]`: Lane index within the warp ($0 \dots 31$).
- `[15:8]`: Warp index within the block (`warp_in_block`, $0 \dots 3$).
- `[23:16]`: Flat thread index within the block (`threadIdx` $= \text{warp\_in\_block} \times 32 + \text{lane}$, range $0 \dots 127$).
- `[31:24]`: Reserved (zero).

#### Identity Selectors (`SIMT_IDENT`, funct3 = 7)
Selected by the 7-bit `funct7` field:
- `funct7 = 0` (`IDENT_BLOCK_IDX`): Returns current `block_id`.
- `funct7 = 1` (`IDENT_BLOCK_DIM`): Returns block thread count (`block_dim`).
- `funct7 = 2` (`IDENT_GRID_DIM`): Returns total blocks in grid (`grid_dim`).
- `funct7 = 3` (`IDENT_HW_TID`): Returns unique physical thread coordinate `hw_tid`.

---

## 5. Physical Thread Indexing & Stack Architecture

### 5.1 Physical Thread Coordinate (`hw_tid`)
In multi-core configurations, each core is assigned a physical core ID (`sm_id`). The unique physical seat index across the entire chip is computed as:

$$\text{hw\_tid} = \big((\text{sm\_id} \times \text{WARPS\_RESIDENT}) + w\big) \times \text{THREADS\_PER\_WARP} + \text{lane}$$

For $N$ cores, `hw_tid` spans the contiguous range $[0, \; 128N - 1]$.

```
Core sm_id = 0 : hw_tid   0 .. 127
Core sm_id = 1 : hw_tid 128 .. 255
Core sm_id = 2 : hw_tid 256 .. 383
Core sm_id = 3 : hw_tid 384 .. 511
```

### 5.2 Private Stack Carving Contract
To prevent stack clobbering among concurrent threads across cores and warps, each physical thread is allotted a dedicated 1 KB stack slice:

$$\text{sp}_{\text{lane}} = \text{stack\_top} - (\text{hw\_tid} + 1) \times \text{STACK\_BYTES\_PER\_THREAD}$$

- Stacks grow downwards from `stack_top`.
- Any external system dispatcher must reserve at least:
  $$\text{Memory Required} = N_{\text{cores}} \times 128 \times 1024 \text{ bytes} \quad (512 \text{ KB for 4 cores})$$
  below `stack_top` solely for SIMT thread stacks.

---

## 6. Block Dispatch & Execution Interface Contract

The IP provides a standardized block launch structure and entry method:

```cpp
struct BlockLaunch {
    reg_t    entry;      // Kernel function entry address (start PC in memory)
    uint32_t block_id;   // Current block index (0 .. grid_dim - 1)
    uint32_t block_dim;  // Active threads in this block (1 .. 128)
    uint32_t grid_dim;   // Total blocks launched in the grid
    reg_t    args;       // Pointer / scalar argument for the kernel
    reg_t    stack_top;  // Base ceiling address of the device stack area
};

uint64_t run_block(const BlockLaunch& b, bool* timed_out = nullptr, 
                   uint64_t max_issues = 1000000);
```

### 6.1 Pre-Execution Initialization (Core-Internal)
When `run_block(b)` is invoked, the core automatically performs the following setup:

1. **Parameter Validation:**  
   Verifies $1 \le \text{block\_dim} \le 128$. If violated, execution immediately aborts with 0 cycles.
2. **Warp Slicing:**  
   Calculates required warps: $N_{\text{warps}} = \lceil \text{block\_dim} / 32 \rceil$.
3. **Warp Slot Allocation & Seeding:**  
   For each warp slot $w \in [0, N_{\text{warps}}-1]$:
   - Sets `warp.pc = b.entry`.
   - Clears `warp.ipdom_stack`.
   - Configures `warp.tmask`:
     - Warps $0 \dots N_{\text{warps}}-2$: Full mask (`0xFFFFFFFF`).
     - Last warp ($N_{\text{warps}}-1$): Masked to $(1 \ll (\text{block\_dim} \pmod{32})) - 1$ if not a multiple of 32.
   - Clears registers $x1 \dots x31$ to `0`.
   - Seeds standard ABI calling registers for every active lane:
     - `x2` (`sp`): Initialized to `lane_stack_top(b, w, lane)`.
     - `x1` (`ra`): Initialized to `BLOCK_RETURN_PC` (`0xFFFFFFF0`).
     - `x10` (`a0`): Initialized to `b.args`.
   - Records metadata: `block_id = b.block_id`, `warp_in_block = w`.

### 6.2 Execution Loop & Completion
- **Sequential Warp Execution:** Resident warps execute to completion sequentially ($w = 0 \dots N_{\text{warps}}-1$).
- **Termination Signal:** A warp is marked halted when `warp.pc == BLOCK_RETURN_PC` (triggered when the C kernel function executes `ret` / `jalr x0, ra, 0`), or when it executes a self-loop.
- **Hang Protection:** If cumulative issued instructions reach `max_issues` (default 1,000,000), execution aborts and sets `*timed_out = true`.
- **Return Value:** Returns total instructions issued across all warps in the block.

---

## 7. System Integration & Dispatch Expectations

Any external subsystem interfacing with `SIMTCore` must adhere to these rules:

1. **Kernel Image & Memory Pre-population:**  
   Kernel binaries must be loaded into shared memory prior to dispatch. `b.entry` must point to word-aligned executable instructions.
2. **Kernel Function Signature:**  
   Kernels must follow standard C function linkage:
   ```c
   void kernel_entry(void* args);
   ```
   Functions must return normally via `ret` (`ra`).
3. **Block Dimension Bounds:**  
   Never pass `block_dim == 0` or `block_dim > 128`.
4. **Stack Region Placement:**  
   `b.stack_top` must be placed with sufficient headroom below any host stack to ensure non-overlapping address spaces.
5. **Unsupported Operations within Launched Kernels:**
   - Multi-warp rendezvous (`BAR`) and dynamic warp allocation (`WSPAWN`) are currently unpopulated stubs and must not be emitted.
   - Dynamic `TMC` mask widening inside divergent blocks is prohibited.
