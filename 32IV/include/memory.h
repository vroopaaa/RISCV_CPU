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

    // Write functions
    void write_byte(uint32_t address, uint8_t data);
    void write_halfword(uint32_t address, uint16_t data);
    void write_word(uint32_t address, uint32_t data);
};

#endif // MEMORY_H