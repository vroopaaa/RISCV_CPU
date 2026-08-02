#include "loader.h"
#include <fstream>
#include <vector>
#include <iostream>

bool load_binary(Memory& memory, const std::string& filepath, uint32_t base_addr) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[Loader Error] Could not open file: " << filepath << "\n";
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "[Loader Error] Failed to read file: " << filepath << "\n";
        return false;
    }

    for (std::streamsize i = 0; i < size; i++) {
        memory.write_byte(base_addr + static_cast<uint32_t>(i), buffer[i]);
    }

    std::cout << "[Loader] Loaded " << size << " bytes from " << filepath
               << " into memory at 0x" << std::hex << base_addr << std::dec << "\n";
    return true;
}