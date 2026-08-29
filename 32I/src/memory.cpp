#include "../include/memory.h"
#include "../include/NPU.h"
#include <iomanip>
#include <fstream>

// Checks whether an address falls inside the NPU's memory-mapped window
static inline bool is_npu_address(uint32_t address) {
    return address >= NPU::NPU_BASE && address < NPU::NPU_BASE + NPU::NPU_WINDOW_SIZE;
}

// Constructor: Initializes the memory array with zeros
Memory::Memory(size_t size) {
    mem_array.resize(size, 0);
}

// Helper: Prevents segmentation faults if the CPU asks for a bad address
bool Memory::check_bounds(uint32_t address, uint32_t size) {
    if (address + size > mem_array.size()) {
        std::cerr << "[Memory Error] Out of bounds access at address: 0x"
                   << std::hex << address << std::dec << "\n";
        return false;
    }
    return true;
}

// ==========================================
// READ FUNCTIONS (Little-Endian)
// ==========================================

uint8_t Memory::read_byte(uint32_t address) {
    if (is_npu_address(address)) {
        std::cerr << "[Memory Error] NPU registers require word-aligned access\n";
        return 0;
    }
    if (!check_bounds(address, 1)) return 0;
    return mem_array[address];
}

uint16_t Memory::read_halfword(uint32_t address) {
    if (is_npu_address(address)) {
        std::cerr << "[Memory Error] NPU registers require word-aligned access\n";
        return 0;
    }
    if (!check_bounds(address, 2)) return 0;
    uint16_t byte0 = mem_array[address];
    uint16_t byte1 = mem_array[address + 1];
    return (byte1 << 8) | byte0;
}

uint32_t Memory::read_word(uint32_t address) {
    if (is_npu_address(address)) return npu.read(address);
    if (!check_bounds(address, 4)) return 0;
    uint32_t byte0 = mem_array[address];
    uint32_t byte1 = mem_array[address + 1];
    uint32_t byte2 = mem_array[address + 2];
    uint32_t byte3 = mem_array[address + 3];
    return (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
}

void Memory::read_block(uint32_t addr, uint8_t* dest, size_t num_bytes) {
    for (size_t i = 0; i < num_bytes; i++) {
        dest[i] = read_byte(addr + i);
    }
}

// ==========================================
// WRITE FUNCTIONS (Little-Endian)
// ==========================================

void Memory::write_byte(uint32_t address, uint8_t data) {
    if (is_npu_address(address)) {
        std::cerr << "[Memory Error] NPU registers require word-aligned access\n";
        return;
    }
    if (!check_bounds(address, 1)) return;
    mem_array[address] = data;
}

void Memory::write_halfword(uint32_t address, uint16_t data) {
    if (is_npu_address(address)) {
        std::cerr << "[Memory Error] NPU registers require word-aligned access\n";
        return;
    }
    if (!check_bounds(address, 2)) return;
    mem_array[address]     = data & 0xFF;
    mem_array[address + 1] = (data >> 8) & 0xFF;
}

void Memory::write_word(uint32_t address, uint32_t data) {
    if (is_npu_address(address)) { npu.write(address, data); return; }
    if (!check_bounds(address, 4)) return;
    mem_array[address]     = data & 0xFF;
    mem_array[address + 1] = (data >> 8) & 0xFF;
    mem_array[address + 2] = (data >> 16) & 0xFF;
    mem_array[address + 3] = (data >> 24) & 0xFF;
}

// Shared formatting for both dump_range overloads below.
static void write_range(Memory& mem, uint32_t start, uint32_t end, std::ostream& out) {
    for (uint32_t addr = start; addr < end; addr += 4) {
        out << "0x" << std::hex << std::setfill('0') << std::setw(8) << addr
            << ": " << std::setw(8) << mem.read_word(addr)
            << std::dec << std::setfill(' ') << "\n";
    }
}

// Prints every word in [start, end) as hex, address-labeled, one per line.
void Memory::dump_range(uint32_t start, uint32_t end) {
    write_range(*this, start, end, std::cout);
}

// Same, written to file_path instead of stdout.
void Memory::dump_range(uint32_t start, uint32_t end, const std::string& file_path) {
    std::ofstream out(file_path);
    if (!out) {
        std::cerr << "[Memory Error] Could not open dump file: " << file_path << "\n";
        return;
    }
    write_range(*this, start, end, out);
}

// Prints a rows x cols grid of signed decimal words starting at base_addr
// (row-major, contiguous) -- e.g. a matrix already assembled in memory by
// the CPU's custom-0 NPU transfer instructions.
void Memory::print_matrix(uint32_t base_addr, uint32_t rows, uint32_t cols) {
    std::cout << std::dec << std::setfill(' ');
    for (uint32_t i = 0; i < rows; i++) {
        std::cout << "  ";
        for (uint32_t j = 0; j < cols; j++)
            std::cout << std::setw(6) << (int32_t)read_word(base_addr + (i * cols + j) * 4);
        std::cout << "\n";
    }
}
