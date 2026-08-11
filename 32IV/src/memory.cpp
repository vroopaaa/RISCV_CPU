#include "../include/memory.h"

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
    if (!check_bounds(address, 1)) return 0;
    return mem_array[address];
}

uint16_t Memory::read_halfword(uint32_t address) {
    if (!check_bounds(address, 2)) return 0;
    
    uint16_t byte0 = mem_array[address];
    uint16_t byte1 = mem_array[address + 1];
    
    return (byte1 << 8) | byte0;
}

uint32_t Memory::read_word(uint32_t address) {
    if (!check_bounds(address, 4)) return 0;
    
    uint32_t byte0 = mem_array[address];
    uint32_t byte1 = mem_array[address + 1];
    uint32_t byte2 = mem_array[address + 2];
    uint32_t byte3 = mem_array[address + 3];
    
    return (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
}

bool Memory::probe(uint32_t address, uint32_t n) const {
    return (uint64_t)address + n <= mem_array.size();
}

std::vector<uint8_t> Memory::read_bytes(uint32_t address, uint32_t n) {
    std::vector<uint8_t> result(n, 0);
    if (!check_bounds(address, n)) return result;

    for (uint32_t i = 0; i < n; i++) {
        result[i] = mem_array[address + i];
    }
    return result;
}

// ==========================================
// WRITE FUNCTIONS (Little-Endian)
// ==========================================

void Memory::write_byte(uint32_t address, uint8_t data) {
    if (!check_bounds(address, 1)) return;
    mem_array[address] = data;
}

void Memory::write_halfword(uint32_t address, uint16_t data) {
    if (!check_bounds(address, 2)) return;
    
    // Extract the lower and upper 8 bits
    mem_array[address]     = data & 0xFF;         // Lowest byte
    mem_array[address + 1] = (data >> 8) & 0xFF;  // Highest byte
}

void Memory::write_word(uint32_t address, uint32_t data) {
    if (!check_bounds(address, 4)) return;
    
    // Break the 32-bit data into 4 individual bytes
    mem_array[address]     = data & 0xFF;         // Lowest byte
    mem_array[address + 1] = (data >> 8) & 0xFF;
    mem_array[address + 2] = (data >> 16) & 0xFF;
    mem_array[address + 3] = (data >> 24) & 0xFF; // Highest byte
}

// Byte-enabled write: bytes with byte_we[i] == false are left completely untouched
// (used by masked vector stores, where inactive lanes must have zero side effect).
void Memory::write_bytes(uint32_t address, const std::vector<uint8_t>& data, const std::vector<bool>& byte_we) {
    uint32_t n = static_cast<uint32_t>(data.size());
    if (!check_bounds(address, n)) return;

    for (uint32_t i = 0; i < n; i++) {
        if (byte_we[i]) {
            mem_array[address + i] = data[i];
        }
    }
}