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
#include "CPU.h"
#include "memory.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cctype>

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
        std::cerr << "usage: trace_harness <bin> <disassembly.txt> <out.txt> [max_cycles] [mode] [issue_width]\n";
        return 1;
    }
    std::string bin_path = argv[1];
    std::string dis_path = argv[2];
    std::string out_path = argv[3];
    long max_cycles = argc > 4 ? std::atol(argv[4]) : 20000000;
    std::string mode = argc > 5 ? argv[5] : "scalar";
    int issue_width = argc > 6 ? std::atoi(argv[6]) : 0;
    bool superscalar = (mode == "superscalar");

    auto dis_map = load_disassembly(dis_path);

    Memory memory(4 * 1024 * 1024);
    if (!load_binary(memory, bin_path, 0x0)) return 1;
    CPU cpu(&memory);
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
            for (int i = 0; i < m; i++) {
                uint32_t slot_pc = base_pc + 4 * i;
                auto it = dis_map.find(slot_pc);
                std::string text = (it != dis_map.end()) ? it->second : "<unknown>";
                if (i > 0) line << " | ";
                line << "0x" << std::hex << slot_pc << std::dec << ": " << text
                     << npu_tag(cpu.issued_opcode(i), cpu.issued_funct3(i));
            }
            out << cycles << "\t0x" << std::hex << base_pc << std::dec << "\t" << m
                << "\t" << line.str() << "\n";
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

            auto it = dis_map.find(pc);
            std::string text = (it != dis_map.end()) ? it->second : "<unknown>";
            out << cycles << "\t0x" << std::hex << pc << std::dec << "\t" << text
                << npu_tag(cpu.last_opcode(), cpu.last_funct3()) << "\n";
            cycles++;
        }
        std::cerr << "Traced " << cycles << " instructions to " << out_path
                   << (cycles >= max_cycles ? " (HIT CYCLE CAP -- did not halt)" : " (halted normally)") << "\n";
    }

    return 0;
}
