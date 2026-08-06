#ifndef LOADER_H
#define LOADER_H

#include <string>
#include "../../include/memory.h"

bool load_binary(Memory& memory, const std::string& filepath, uint32_t base_addr = 0x0);

#endif