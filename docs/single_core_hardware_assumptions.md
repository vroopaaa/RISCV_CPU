# Single-Core Hardware Architecture & Multi-Core Scaling Roadmap

> **Target Platform:** RSVP RISC-V Emulator (`32I`, `32IV`, `NPU`)  
> **Topic:** Comprehensive single-core hardware assumptions, block diagrams, caches, ALU, NPU memory, and 2-to-4 core multi-core architecture.

---

## 1. Executive Summary & Top-Level Architecture

Before scaling RSVP from a single core to a **2-core** and **4-core** symmetric multiprocessor (SMP) SoC, we must clearly audit the **hardware blocks** and **architectural assumptions** present in the current codebase.

In the current implementation:
1. **Core Compute Pipeline:** Modeled as an in-order superscalar core (issue width $n \in [1, 10]$) with a 128-entry Branch Target Buffer (BTB) using 2-bit saturating counters for branch speculation.
2. **Caches:** **There are currently NO caches** (no L1-I, no L1-D, and no L2/L3). All memory accesses directly hit a flat 4 MB `std::vector<uint8_t>` main memory with **zero wait-states** and **infinite bandwidth**.
3. **Execution & Functional Units:** All instructions (including RV32M hardware multiply and divide) have an assumed **uniform 1-cycle latency**. There is no register forwarding; RAW hazards stall dependent instructions until the next cycle.
4. **NPU Matrix Accelerator:** Configurable $16 \times 16$ systolic matrix engine with 3 KB of internal SRAM (Tile A, Tile B, Tile C). Computes a full $16 \times 16 \times 16$ (4,096 MACs) **instantaneously in zero simulation cycles**, and moves 1,024-byte tiles via synchronous Custom-0 DMA instructions in a single cycle.

```
+---------------------------------------------------------------------------------------------------+
|                                      SINGLE-CORE RSVP SoC                                         |
|                                                                                                   |
|  +------------------------------------ CORE (Hart 0) ------------------------------------------+  |
|  |                                                                                            |  |
|  |   +---------------------+        +--------------------+        +-----------------------+   |  |
|  |   | Instruction Fetch   | =====> | Decode & Hazard    | =====> | Execution & RegFile   |   |  |
|  |   |                     |        |                    |        |                       |   |  |
|  |   | - PC & Next PC Gen  |        | - Parallel Decode  |        | - 32 x 32-bit GPRs    |   |  |
|  |   | - 128-entry BTB     |        | - RAW Hazard Scan  |        | - ALU Array (RV32I)   |   |  |
|  |   | - 2-bit Sat Counter |        | - MemClass Check   |        | - RV32M Mul/Div       |   |  |
|  |   | - Speculative Fetch |        | - Issue Prefix (m) |        | - Branch Target Eval  |   |  |
|  |   +---------------------+        +--------------------+        +-----------------------+   |  |
|  |              |                                                             |               |  |
|  +--------------|-------------------------------------------------------------|---------------+  |
|                 | (Fetch Bus: n words/cycle)                                  | (Load/Store)     |
|                 v                                                             v                  |
|  +---------------------------------------------------------------------------------------------+  |
|  |                      UNCACHED PHYSICAL MEMORY INTERFACE / BUS                               |  |
|  |                (Assumes 0-cycle latency, infinite throughput, no arbitration)               |  |
|  +---------------------------------------------------------------------------------------------+  |
|                 |                                                             |                  |
|                 v                                                             v                  |
|  +--------------------------------------+                   +----------------------------------+  |
|  |          MAIN MEMORY (RAM)           |                   |    NPU ACCELERATOR (0x80000000)  |  |
|  |                                      |                   |                                  |  |
|  |  - 4 MB std::vector<uint8_t>         | <================ |  - 3 KB Internal SRAM (A, B, C)  |  |
|  |  - Flat, Little-Endian               |  (Custom-0 DMA:   |  - MMIO Regs (Dims, Trig, Mac)   |  |
|  |  - Instant word/block read/write     |   1024 B / cycle) |  - 16x16x16 Systolic Matrix Core |  |
|  +--------------------------------------+                   +----------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Core Pipeline Blocks & Hardware Assumptions

### 2.1 Instruction Fetch & Branch Prediction (`fetch_n`)
* **Fetch Engine:** Fetches up to `issueWidth` ($n \le 10$) consecutive 32-bit instruction words from memory starting at `pc`.
* **Branch Target Buffer (BTB):**
  * Size: 128 entries, direct-mapped, indexed by `(addr >> 2) & 0x7F`.
  * State: 2-bit saturating counter (`00` = Strongly Not-Taken, `01` = Weakly Not-Taken, `10` = Weakly Taken, `11` = Strongly Taken).
  * Target address recorded in `target_pc`.
* **Speculation Mechanics:**
  * If a branch instruction is encountered and BTB indicates `state >= 2` (Taken), the frontend **speculatively redirects fetch in the middle of the fetch packet**: subsequent slots in the same cycle are fetched from `target_pc`.
  * If the branch is later resolved as not-taken (or vice-versa), downstream speculative instructions in the packet are squashed (`windowCancelled[tail] = true`).
  * Non-speculative Jumps (`JAL`, `JALR`) terminate the fetch packet immediately; no speculation is performed across indirect jumps.
* **Underlying Hardware Assumptions:**
  1. *Multi-word single-cycle fetch:* Core reads up to 40 contiguous bytes simultaneously in 1 cycle.
  2. *Zero instruction fetch latency:* No instruction cache line fill delay or bus wait-states.
  3. *Zero-cycle branch redirect:* Branch target redirect happens within the same cycle's packet without an extra bubble cycle.

### 2.2 Decode & Hazard Detection (`decode_all`, `hazard_scan`)
* **Decode Unit:** Replicates decoding logic across all $n$ fetched slots. Extracts standard RISC-V fields (`opcode`, `rd`, `rs1`, `rs2`, `funct3`, `funct7`) and immediate generators (U, I, S, B, J, Z types).
* **In-Order Superscalar Hazard Scanner:**
  * **RAW (Read-After-Write):** If instruction $i$ reads a register written by an earlier instruction $j < i$ in the current window, issue is truncated at $i$. **No register forwarding** is modeled.
  * **Structural Memory Hazards (`MemClass`):**
    * Classes: `MEM_NONE`, `MEM_BANK_A`, `MEM_BANK_B`, `MEM_STORE`.
    * Two stores conflict (single external memory write port).
    * Two loads into the *same* NPU bank conflict (single internal bank write port).
    * A store and an NPU bank load **do not conflict** (different physical ports).
    * ALU instructions (`MEM_NONE`) never conflict with memory operations.
  * **WAR / WAW:** Automatically preserved because all register reads occur before any register writes, and commit (`writeback_m`) writes registers in strict program order.

### 2.3 Execution Units (ALU & RV32M)
* **Functional Units:**
  * RV32I integer arithmetic, logic, shifts, and comparisons (`ADD`, `SUB`, `AND`, `OR`, `XOR`, `SLL`, `SRL`, `SRA`, `SLT`, `SLTU`).
  * Upper immediate operations (`LUI`, `AUIPC`).
  * RV32M extension (`MUL`, `MULH`, `MULHSU`, `MULHU`, `DIV`, `DIVU`, `REM`, `REMU`).
* **Underlying Hardware Assumptions:**
  1. *Uniform 1-Cycle Latency:* Every instruction, including 32-bit hardware division and multiplication, executes in a single clock cycle.
  2. *Symmetric ALUs:* Up to $n$ ALUs exist in parallel; any slot in the superscalar window can execute any arithmetic instruction simultaneously.

---

## 3. The Memory Subsystem: Caches, Bandwidth & Latency

### 3.1 Current Reality: No Caches Exist
A common misconception when inspecting high-level simulators is assuming an L1 cache hierarchy is operating underneath. In RSVP:
* **L1 Instruction Cache (L1-I):** **None.** Program Counter fetches directly invoke `Memory::read_word()`.
* **L1 Data Cache (L1-D):** **None.** Scalar loads (`LW`, `LH`, `LB`) and stores (`SW`, `SH`, `SB`) directly mutate the `std::vector<uint8_t>` array.
* **L2 / L3 Shared Caches:** **None.**

### 3.2 Key Memory Assumptions
1. **Zero-Latency Memory Access:** In physical hardware, accessing on-chip SRAM takes 1–3 cycles, while off-chip DRAM takes 50–150 ns (100+ cycles). RSVP models memory as a synchronous C++ array index operation with **0 wait cycles**.
2. **Infinite Port Concurrency / Bandwidth:** Within a single cycle, the CPU can:
   - Fetch up to 10 instruction words (40 bytes),
   - Issue multiple data memory loads or stores, and
   - Perform a 1,024-byte bulk tile DMA transfer via Custom-0,
   all hitting the same `Memory` object without any bus arbitration or bandwidth contention.
3. **Absence of Memory Coherence:** Because memory is a single monolithic array and there are no private caches, there is currently no concept of cache lines, dirty lines, write buffers, or coherence invalidation.

---

## 4. NPU Matrix Accelerator Architecture

The NPU is a hardware matrix-multiply engine integrated into the memory map and CPU ISA.

### 4.1 Internal Storage & MMIO Map
* **Base Address:** `0x80000000`
* **Internal SRAM:** 3 KB contiguous data window:
  - `0x80000100` – `0x800004FF`: **Matrix A Buffer** (256 words = 16x16 int32 = 1,024 bytes)
  - `0x80000500` – `0x800008FF`: **Matrix B Buffer** (256 words = 16x16 int32 = 1,024 bytes)
  - `0x80000900` – `0x80000CFF`: **Matrix C / Accumulator Buffer** (256 words = 16x16 int32 = 1,024 bytes)
* **MMIO Control Registers:**
  - `DIM_M`, `DIM_K`, `DIM_N`: Dimension configuration (clamped to `MAX_DIM = 16`).
  - `TRIGGER_ADDR` (`0x8000000C`): Triggers one-shot $C = A \times B$.
  - `MAC_ADDR` (`0x80000010`): Triggers multiply-accumulate $C = C + (A \times B)$.
  - `RESET_ADDR` (`0x80000014`): Clears buffers and done bit (leaves dimensions intact).
  - `STATUS_ADDR` (`0x80000018`): Returns status (bit 0 = `DONE`).

### 4.2 Custom-0 RISC-V Instructions (Opcode `0x0B`)
To eliminate the overhead of looping through 256 individual `SW` instructions to load a tile, RSVP defines custom hardware instructions executed in `read()` / `read_m()`:
* **`funct3 = 0` (Load Tile A):** Copies 256 contiguous words (1,024 bytes) from address `rs1` in main memory into NPU Matrix A SRAM.
* **`funct3 = 1` (Load Tile B):** Copies 256 contiguous words from `rs1` into NPU Matrix B SRAM.
* **`funct3 = 2` (Store Tile C):** Performs a 2D strided store: writes the 16x16 result tile from NPU Matrix C SRAM back to main memory starting at `rs1` with row stride `rs2` bytes.
* **`funct3 = 3` (Debug Matrix Print):** Outputs an $N \times N$ matrix to stdout.

### 4.3 Hardware Assumptions in the NPU Model
1. **Instantaneous Compute:** A full $16 \times 16 \times 16$ matrix multiplication (4,096 multiply-accumulates) completes in **0 simulation cycles**. Writing to `TRIGGER` or `MAC` finishes synchronously before the next instruction executes.
2. **Instantaneous Tile DMA:** The 1,024-byte bulk transfer loops execute synchronously inside the CPU stage in zero cycles.
3. **Single Core Coupling:** The NPU is assumed to be directly attached to Hart 0; there are currently no access control locks or multi-master arbitration logic.

---

## 5. Architectural Comparison: Simulator vs. Physical Silicon

| Subsystem / Feature | Current RSVP Simulator | Real Physical Silicon (e.g. SiFive / Rocket / BOOM) |
| :--- | :--- | :--- |
| **L1 Instruction Cache** | **None** (Direct array read) | 16 KB – 64 KB, 2–4 way set associative, 64B cache lines |
| **L1 Data Cache** | **None** (Direct array read/write) | 16 KB – 64 KB, 4–8 way set associative, write-back |
| **Memory Latency** | **0 cycles** (Instantaneous) | L1: 1–3 cycles; L2: 10–20 cycles; DRAM: 100–250 cycles |
| **Memory Bandwidth** | **Infinite** (Up to 1040 bytes/cycle) | 64-bit or 128-bit AXI/TileLink bus (8–16 bytes/cycle) |
| **ALU Multiplier** | 1 cycle uniform | 2–4 cycles pipelined |
| **ALU Divider** | 1 cycle uniform | 16–34 cycles iterative non-blocking state machine |
| **NPU Compute Latency** | 0 cycles (Instantaneous) | Pipelined systolic array: 16–32 cycles per tile |
| **NPU Tile Transfer** | 1 cycle synchronous copy | Asynchronous DMA controller with interrupt completion |
| **Atomic Operations** | Not implemented | RV32A extension (`LR.W`, `SC.W`, `AMOADD`, `AMOSWAP`) |
| **Inter-Core Signaling** | None | CLINT (Core Local Interruptor), MSIP (Software Interrupts) |

---

## 6. Scaling to Multi-Core: 2-Core & 4-Core Roadmap

When transitioning from single-core to 2-core and 4-core execution, several foundational hardware assumptions will break and must be redesigned.

```
                      +-------------------------------------------------------------+
                      |                 4-CORE MULTI-CORE TOPOLOGY                  |
                      +-------------------------------------------------------------+
                            |                   |                   |                   |
                     +--------------+    +--------------+    +--------------+    +--------------+
                     | Core 0 (H0)  |    | Core 1 (H1)  |    | Core 2 (H2)  |    | Core 3 (H3)  |
                     | - PC, Regs   |    | - PC, Regs   |    | - PC, Regs   |    | - PC, Regs   |
                     | - BTB (128)  |    | - BTB (128)  |    | - BTB (128)  |    | - BTB (128)  |
                     | - Superscalar|    | - Superscalar|    | - Superscalar|    | - Superscalar|
                     +--------------+    +--------------+    +--------------+    +--------------+
                            |                   |                   |                   |
                            +-------------------+---------+---------+-------------------+
                                                          |
                                                          v
                                     +------------------------------------------+
                                     |    SYSTEM INTERCONNECT / BUS ARBITER     |
                                     |     (Round-Robin / Priority Crossbar)    |
                                     +------------------------------------------+
                                           |                             |
                                           v                             v
                        +------------------------------------+  +--------------------------------+
                        |     SHARED MEMORY SUBSYSTEM        |  |     NPU ACCELERATOR CLUSTER    |
                        |                                    |  |                                |
                        | - Multi-Banked RAM or Arbiter      |  | Option A: 1 Shared NPU + Mutex |
                        | - Atomic Ops (LR/SC, AMO)          |  | Option B: 4 Private NPUs       |
                        | - Independent Stack per Hart       |  |                                |
                        +------------------------------------+  +--------------------------------+
```

### 6.1 Architectural Milestones for 2-Core & 4-Core

#### 1. Hart Identification (`mhartid` CSR)
Each core instance must know its identity ($0, 1, 2, 3$).
* When software boots, it reads `csrr a0, mhartid`.
* Core 0 proceeds with global initialization (setting up BSS, data segment, NPU base configuration).
* Cores 1..3 initialize their own stack pointers (`sp = stack_top - hartid * stack_size`) and wait for work.

#### 2. Interconnect & Bus Arbitration
In a single core, `Memory` has only one caller. With multiple cores:
* If both Core 0 and Core 1 attempt to issue loads or stores in the same simulation cycle, access must be arbitrated.
* **Phase 1 Approach (Functional):** Cores share the existing `Memory` object, but accesses are stepped sequentially or interleaved round-robin.
* **Phase 2 Approach (Realistic):** Model memory contention stalls or separate memory banks to observe multi-core bus pressure.

#### 3. Hardware Synchronization & Atomics (RV32A)
Without atomic instructions, multiple cores cannot safely implement mutexes, work queues, or barrier synchronization.
* Minimal RV32A subset required:
  * `LR.W` (Load-Reserved) & `SC.W` (Store-Conditional), OR
  * `AMOSWAP.W` / `AMOADD.W` (Atomic Memory Operations).
* Alternatively, a memory-mapped hardware spinlock/mutex peripheral can be provided in the memory map.

#### 4. The NPU Topology Decision
How should the NPU accelerator be allocated among 2 or 4 cores?
* **Option A: Single Shared NPU with Hardware Mutex**
  * One physical NPU accelerator at `0x80000000`.
  * Cores acquire a hardware lock register before setting dimensions and loading tiles.
  * *Advantage:* Minimal area overhead; mimics a shared SoC neural coprocessor.
  * *Disadvantage:* Cores contend for the NPU; serialization bottleneck.
* **Option B: Core-Private NPUs (NPU per Hart)**
  * Each core owns its own NPU accelerator (e.g. Hart 0 at `0x80000000`, Hart 1 at `0x80010000`, Hart 2 at `0x80020000`, Hart 3 at `0x80030000`).
  * *Advantage:* Full embarrassingly-parallel throughput; no lock contention; Custom-0 instructions cleanly route to the local NPU.
  * *Disadvantage:* Replicates SRAM and compute hardware $4\times$.

#### 5. Cache Strategy (Uncached vs. Private L1s)
* **Recommendation:** Start with **Uncached Shared Memory** first (Phase 1).
  * In Phase 1, maintain the flat, uncached `Memory` array so that all cores immediately see each other's writes (Sequential Consistency).
  * This allows focusing on multi-core execution correctness, thread dispatch, and matmul workload partitioning without the complexity of cache snooping (MESI) protocols.
  * Once the multi-core software pipeline is proven correct, add private L1 caches and coherence protocols as an architectural study.

#### 6. Simulation Stepping & Halt Semantics
* The simulation driver must loop through all active cores per cycle:
  ```cpp
  for (int cycle = 0; cycle < max_cycles; cycle++) {
      for (int h = 0; h < num_cores; h++) {
          if (!cores[h]->is_halted()) {
              cores[h]->step_cycle();
          }
      }
      if (all_cores_halted()) break;
  }
  ```
* A single core executing a self-loop (e.g. `_end: j _end`) must only halt *itself*, not terminate the whole simulation.

---

## 7. Key Discussion Points for the User

1. **NPU Allocation:** Do you prefer **1 Shared NPU** (with synchronization locks) or **Private NPUs** (1 per core, allowing concurrent $16 \times 16$ tile execution)?
2. **Synchronization Mechanism:** Should we implement the **RV32A extension** (`LR.W`/`SC.W` and `AMO`), or start with a simple memory-mapped mutex/flag peripheral?
3. **Cache Policy:** Do we proceed with **uncached shared memory** for initial functional 2-core bringup, or do you want to introduce **L1 Instruction & Data caches** right now?
4. **Target Workload:** Will the initial 2-core test be a partitioned parallel matrix multiplication (e.g. Core 0 computes top half, Core 1 computes bottom half), or independent parallel benchmarks?
