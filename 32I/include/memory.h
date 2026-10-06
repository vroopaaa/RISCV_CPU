#ifndef MEMORY_H
#define MEMORY_H

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include "NPU.h"

class Memory {
private:
    // The physical storage array representing byte-addressable memory
    std::vector<uint8_t> mem_array;
    
    //NPU object
    NPU npu;
    // Helper function to check if an address is valid
    bool check_bounds(uint32_t address, uint32_t size);

public:
    // Constructor initializes the memory size
    Memory(size_t size);
    size_t size() const { return mem_array.size(); } 

    // Read functions
    uint8_t  read_byte(uint32_t address);
    uint16_t read_halfword(uint32_t address);
    uint32_t read_word(uint32_t address);
    void read_block(uint32_t addr, uint8_t* dest, size_t num_bytes);

    // Write functions
    void write_byte(uint32_t address, uint8_t data);
    void write_halfword(uint32_t address, uint16_t data);
    void write_word(uint32_t address, uint32_t data);

    // Prints every word in [start, end) as hex, one per line, address-labeled.
    void dump_range(uint32_t start, uint32_t end);
    // Same, written to file_path instead of stdout -- for harnesses that need
    // to read the region back programmatically (e.g. a Python test driver)
    // rather than scrape it out of interleaved stdout.
    void dump_range(uint32_t start, uint32_t end, const std::string& file_path);
    // Prints a rows x cols grid of signed decimal words starting at
    // base_addr (row-major, contiguous) -- unlike dump_range, this reads
    // actual matrix content (e.g. a result matrix already stored by NPU
    // bulk transfers) rather than a raw hex/address listing.
    void print_matrix(uint32_t base_addr, uint32_t rows, uint32_t cols);
};

#endif // MEMORY_H