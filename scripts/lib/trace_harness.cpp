// Dynamic execution tracer, built and run by `assemblyinstruction -trace`.
//
// Steps a CPU+Memory exactly like tests/basic/main.cpp's harness loop, but
// instead of just printing final register state, writes one line per cycle
// EXECUTED (real runtime order, loops repeated as many times as they
// actually ran) -- cycle number, PC, and the disassembled mnemonic, looked
// up from a `riscv64-unknown-elf-objdump -d` listing of the same binary (a
// static disassembly is a reliable 1:1 address->mnemonic map, so there's no
// need for this file to disassemble anything itself).
//
// In scalar mode (default) that's one instruction per line. In superscalar
// mode (see docs/superscalar.md) a cycle can issue several instructions
// together, so all of them go on the SAME line -- that's the point of the
// trace: seeing which instructions the hazard scan actually grouped.
//
// Custom-0 (opcode 0x0B) instructions get an inline NPU tag when present,
// so they're easy to grep out of an otherwise scalar-instruction-heavy
// trace; this is a no-op annotation on binaries that never use that opcode.
//
// GPU programs (LAUNCH, opcode 0x5B): the GPU's work is printed nested right
// under the host's LAUNCH line, then the host trace carries on --
//     57  0xfc  LAUNCH vec_add 5 blocks x 40 threads args=0x2a0
//       +- GPU  sm  blk  warp  sm_cycle  pc  mask  instruction
//       |   0   0    0     0         0x10  ffffffff  SIMT_BLOCK_IDX a5
//       |   block 0 on SM 0: 2 warps, 46 issued
//       +- launch done: device cycles 92 (busiest: SM 0), host resumes
// Blocks appear in the order the emulator ran them; sm_cycle is each SM's own
// count (SMs run in parallel in hardware, so every SM starts at 0), and mask
// is the set of lanes that executed the instruction. Gpu mode "summary"
// keeps only the per-block lines. Custom SIMT / LAUNCH instructions are
// printed by name instead of objdump's raw ".insn 4, 0x...".
#include "CPU.h"
#include "GridLauncher.h"
#include "memory.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cctype>
#include <iomanip>

static bool load_binary(Memory& memory, const std::string& filepath, uint32_t base_addr) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Could not open " << filepath << "\n"; return false; }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    for (std::streamsize i = 0; i < size; i++)
        memory.write_byte(base_addr + (uint32_t)i, buffer[i]);
    return true;
}

// Parses `riscv64-unknown-elf-objdump -d` output: lines like
// "  40:	fe010113          	addi	sp,sp,-32" -> {0x40: "addi sp,sp,-32"}
// Symbol-table header lines ("00000040 <npu_load_a>:") are skipped since
// their pre-colon text isn't pure hex once the "<name>" part is included.
static std::unordered_map<uint32_t, std::string> load_disassembly(const std::string& path) {
    std::unordered_map<uint32_t, std::string> map;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) continue;
        std::string addr_str = line.substr(0, colon);
        bool is_addr = !addr_str.empty();
        for (char c : addr_str)
            if (!isxdigit((unsigned char)c) && c != ' ') { is_addr = false; break; }
        if (!is_addr) continue;
        uint32_t addr;
        try { addr = std::stoul(addr_str, nullptr, 16); } catch (...) { continue; }

        size_t tab1 = line.find('\t', colon);
        if (tab1 == std::string::npos) continue;
        size_t tab2 = line.find('\t', tab1 + 1);
        if (tab2 == std::string::npos) continue;
        std::string text = line.substr(tab2 + 1);
        for (char& c : text) if (c == '\t') c = ' ';
        map[addr] = text;
    }
    return map;
}

// Symbol header lines ("00000010 <vec_add>:") -> {0x10: "vec_add"}.
static std::unordered_map<uint32_t, std::string> load_symbols(const std::string& path) {
    std::unordered_map<uint32_t, std::string> map;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        size_t lt = line.find(" <");
        size_t gt = line.find(">:");
        if (lt == std::string::npos || gt == std::string::npos || gt < lt) continue;
        uint32_t addr;
        try { addr = std::stoul(line.substr(0, lt), nullptr, 16); } catch (...) { continue; }
        map[addr] = line.substr(lt + 2, gt - lt - 2);
    }
    return map;
}

static const char* REG_NAMES[32] = {
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"};

// Name + operands for the custom SIMT (0x2B) and LAUNCH (0x5B) instructions,
// "" for anything else (those keep objdump's text).
static std::string custom_label(uint32_t word) {
    uint32_t opcode = word & 0x7F;
    uint32_t rd = (word >> 7) & 0x1F, funct3 = (word >> 12) & 0x7;
    uint32_t rs1 = (word >> 15) & 0x1F, rs2 = (word >> 20) & 0x1F, funct7 = word >> 25;
    std::string d = REG_NAMES[rd], a = REG_NAMES[rs1], b = REG_NAMES[rs2];
    if (opcode == 0x5B) return (funct3 == 0) ? "LAUNCH " + a + "," + b : "LAUNCH? (reserved funct3)";
    if (opcode != 0x2B) return "";
    switch (funct3) {
        case 0: return "SIMT_TMC " + a;
        case 1: return "SIMT_WSPAWN " + a + "," + b;
        case 2: return "SIMT_TID " + d;
        case 3: return "SIMT_BAR " + a + "," + b;
        case 4: return "SIMT_SPLIT " + a;
        case 5: return "SIMT_JOIN";
        case 6: return "SIMT_PRED " + a;
        default:
            switch (funct7) {
                case 0: return "SIMT_BLOCK_IDX " + d;
                case 1: return "SIMT_BLOCK_DIM " + d;
                case 2: return "SIMT_GRID_DIM " + d;
                case 3: return "SIMT_HW_TID " + d;
                default: return "SIMT_IDENT? " + d;
            }
    }
}

static std::string text_at(const std::unordered_map<uint32_t, std::string>& dis_map, uint32_t pc, uint32_t word) {
    std::string label = custom_label(word);
    if (!label.empty()) return label;
    auto it = dis_map.find(pc);
    return (it != dis_map.end()) ? it->second : "<unknown>";
}

// GPU lines are collected while the LAUNCH runs (inside cpu.read()) and
// written out after the host's LAUNCH line.
struct GpuTrace {
    const std::unordered_map<uint32_t, std::string>* dis_map;
    bool full;
    std::ostringstream buf;
};

static void trace_issue(void* ctx, const SIMTCore::TraceEvent& e) {
    GpuTrace* t = (GpuTrace*)ctx;
    if (!t->full) return;
    t->buf << "  |   " << e.sm << "\t" << e.block << "\t" << e.warp << "\t" << e.sm_cycle
           << "\t0x" << std::hex << e.pc << "\t" << std::setw(8) << std::setfill('0') << e.tmask
           << std::setfill(' ') << std::dec << "\t" << text_at(*t->dis_map, e.pc, e.word) << "\n";
}

static void trace_block(void* ctx, int sm, uint32_t block, uint32_t warps, uint64_t issued) {
    GpuTrace* t = (GpuTrace*)ctx;
    t->buf << "  |   block " << block << " on SM " << sm << ": " << warps << " warps, " << issued << " issued\n";
}

static std::string launch_text(const GridLauncher& gpu, const std::unordered_map<uint32_t, std::string>& symbols) {
    auto sym = symbols.find(gpu.last_entry());
    std::ostringstream l;
    l << "LAUNCH " << (sym != symbols.end() ? sym->second : "<kernel>") << " "
      << gpu.last_num_blocks() << " blocks x " << gpu.last_threads_per_block()
      << " threads args=0x" << std::hex << gpu.last_args() << std::dec;
    return l.str();
}

// The nested GPU part written under the host line that issued the LAUNCH.
static void write_gpu_section(std::ostream& out, const GridLauncher& gpu, GpuTrace& t) {
    if (t.full) out << "  +- GPU  sm\tblk\twarp\tsm_cycle\tpc\tmask\tinstruction\n";
    out << t.buf.str();
    uint32_t busiest = 0;
    for (uint32_t sm = 1; sm < NUM_SMS; sm++)
        if (gpu.sm_cycles((int)sm) > gpu.sm_cycles((int)busiest)) busiest = sm;
    out << "  +- launch " << (gpu.faulted() ? "FAULTED" : gpu.timed_out() ? "TIMED OUT" : "done")
        << ": device cycles " << gpu.sm_cycles((int)busiest) << " (busiest: SM " << busiest << "), "
        << ((gpu.faulted() || gpu.timed_out()) ? "host halts" : "host resumes") << "\n";
}

static const char* npu_tag(uint8_t opcode, uint8_t funct3) {
    if (opcode != 0x0B) return "";
    switch (funct3) {
        case 0: return "   <-- NPU LOAD A (contiguous 256-word)";
        case 1: return "   <-- NPU LOAD B (contiguous 256-word)";
        case 2: return "   <-- NPU STORE C (contiguous 256-word)";
        case 3: return "   <-- NPU PRINT (debug)";
        default: return "   <-- NPU (?)";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: trace_harness <bin> <disassembly.txt> <out.txt> [max_cycles] [mode] [issue_width] [gpu: full|summary]\n";
        return 1;
    }
    std::string bin_path = argv[1];
    std::string dis_path = argv[2];
    std::string out_path = argv[3];
    long max_cycles = argc > 4 ? std::atol(argv[4]) : 20000000;
    std::string mode = argc > 5 ? argv[5] : "scalar";
    int issue_width = argc > 6 ? std::atoi(argv[6]) : 0;
    bool superscalar = (mode == "superscalar");
    bool gpu_full = !(argc > 7 && std::string(argv[7]) == "summary");

    auto dis_map = load_disassembly(dis_path);
    auto symbols = load_symbols(dis_path);

    Memory memory(4 * 1024 * 1024);
    if (!load_binary(memory, bin_path, 0x0)) return 1;
    CPU cpu(&memory);
    // Same GPU as tests/basic/main.cpp (emul), with tracing hooked in.
    GridLauncher gpu(&memory, CPU::HOST_STACK_RESERVE);
    cpu.attach_gpu(&gpu);
    GpuTrace gpu_trace;
    gpu_trace.dis_map = &dis_map;
    gpu_trace.full = gpu_full;
    gpu.set_trace(trace_issue, trace_block, &gpu_trace);
    if (superscalar && issue_width > 0) cpu.set_issue_width(issue_width);

    std::ofstream out(out_path);
    long cycles = 0;
    long total_instructions = 0;

    if (superscalar) {
        // One line per superscalar CYCLE, not per instruction -- everything
        // hazard_scan() issued together this cycle is joined onto that line
        // with " | ", each still tagged with its own PC (they're 4 bytes
        // apart, not the same address) and NPU tag if applicable.
        out << "cycle\tbase_pc\tissued\tinstructions\n";
        while (!cpu.is_halted() && cycles < max_cycles) {
            cpu.fetch_n();
            uint32_t base_pc = cpu.current_pc();
            cpu.decode_all();
            cpu.hazard_scan();
            cpu.execute_m();
            cpu.read_m();
            cpu.writeback_m();

            int m = cpu.last_issue_count();
            std::ostringstream line;
            bool launched = false;
            for (int i = 0; i < m; i++) {
                uint32_t slot_pc = cpu.slot_pc(i);
                std::string text = text_at(dis_map, slot_pc, memory.read_word(slot_pc));
                if (cpu.issued_opcode(i) == CPU::OPCODE_LAUNCH && cpu.issued_funct3(i) == 0) {
                    if (gpu.last_num_blocks() > 0) { text = launch_text(gpu, symbols); launched = true; }
                    else text += "   <-- LAUNCH refused (see stderr)";
                }
                if (i > 0) line << " | ";
                line << "0x" << std::hex << slot_pc << std::dec << ": " << text
                     << npu_tag(cpu.issued_opcode(i), cpu.issued_funct3(i));
                if (cpu.is_cancelled(i)) {
                    line << " [SQUASHED]";
                }
            }
            out << cycles << "\t0x" << std::hex << base_pc << std::dec << "\t" << m
                << "\t" << line.str() << "\n";
            if (launched) write_gpu_section(out, gpu, gpu_trace);  // LAUNCH issues alone, so it's the only slot
            gpu_trace.buf.str("");
            cycles++;
            total_instructions += m;
        }
        std::cerr << "Traced " << cycles << " superscalar cycles (" << total_instructions
                   << " instructions, issue width " << cpu.issue_width() << ") to " << out_path
                   << (cycles >= max_cycles ? " (HIT CYCLE CAP -- did not halt)" : " (halted normally)") << "\n";
    } else {
        out << "cycle\tpc\tinstruction\n";
        while (!cpu.is_halted() && cycles < max_cycles) {
            cpu.fetch();
            uint32_t pc = cpu.current_pc();
            cpu.decode();
            cpu.execute();
            cpu.read();
            cpu.writeback();

            uint32_t word = memory.read_word(pc);
            std::string text = text_at(dis_map, pc, word);
            bool launched = (cpu.last_opcode() == CPU::OPCODE_LAUNCH && cpu.last_funct3() == 0 &&
                             gpu.last_num_blocks() > 0);
            if (launched) {
                text = launch_text(gpu, symbols);
            } else if (cpu.last_opcode() == CPU::OPCODE_LAUNCH) {
                text += "   <-- LAUNCH refused (see stderr)";
            }
            out << cycles << "\t0x" << std::hex << pc << std::dec << "\t" << text
                << npu_tag(cpu.last_opcode(), cpu.last_funct3()) << "\n";
            if (launched) write_gpu_section(out, gpu, gpu_trace);
            gpu_trace.buf.str("");
            cycles++;
        }
        std::cerr << "Traced " << cycles << " instructions to " << out_path
                   << (cycles >= max_cycles ? " (HIT CYCLE CAP -- did not halt)" : " (halted normally)") << "\n";
    }

    return 0;
}
