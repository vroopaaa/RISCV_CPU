#ifndef MEMORY_H
#define MEMORY_H

#include <cstdint>
#include <vector>
#include <iostream>

class Memory {
private:
    // The physical storage array representing byte-addressable memory
    std::vector<uint8_t> mem_array;
    
    // Helper function to check if an address is valid
    bool check_bounds(uint32_t address, uint32_t size);

public:
    // Constructor initializes the memory size
    Memory(size_t size);

    // Read functions
    uint8_t  read_byte(uint32_t address);
    uint16_t read_halfword(uint32_t address);
    uint32_t read_word(uint32_t address);
    // Wide read for vector loads: returns n bytes starting at address
    std::vector<uint8_t> read_bytes(uint32_t address, uint32_t n);
    // Silent bounds check (no error logged) - used by fault-only-first loads, which
    // legitimately probe past the end of valid data as their normal mode of operation.
    bool probe(uint32_t address, uint32_t n) const;

    // Write functions
    void write_byte(uint32_t address, uint8_t data);
    void write_halfword(uint32_t address, uint16_t data);
    void write_word(uint32_t address, uint32_t data);
    // Byte-enabled wide write for vector stores: only bytes with byte_we[i] set are written
    void write_bytes(uint32_t address, const std::vector<uint8_t>& data, const std::vector<bool>& byte_we);
};

#endif // MEMORY_H