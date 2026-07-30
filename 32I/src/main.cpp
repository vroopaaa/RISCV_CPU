#include <iostream>
#include "../include/CPU.h"
#include "../include/memory.h" // Assuming you have a Memory header


int main() {
    std::cout << "Starting RISC-V CPU Emulator..." << std::endl;
    // 1. Instantiate the Memory
    Memory memory(1024); // 1KB of memory for this example      

    // 2. Load the program into memory (starting at PC = 0x0)
    // We use your memory's write_word function to insert the 32-bit instructions
    memory.write_word(0x0, 0x00500093); // ADDI x1, x0, 5
    memory.write_word(0x4, 0x00A08113); // ADDI x2, x1, 10
    
    // (Optional) Add a dummy instruction or a jump to itself at the end 
    // to act as a "halt" state so the CPU doesn't read garbage memory.
    // BEQ x0, x0, 0 (Infinite loop at current PC) -> 0x00000063
    memory.write_word(0x8, 0x00000063); 

    // 3. Instantiate the CPU and pass the memory pointer
    CPU cpu(&memory);

    // 4. Print initial state
    std::cout << "\n--- Initial State ---" << std::endl;
    cpu.print_state();

    // 5. Run the CPU for a few cycles (Tick loop)
    // We will run it for 2 cycles to execute our 2 ADDI instructions
    int num_cycles = 2;
    for (int i = 0; i < num_cycles; i++) {
        std::cout << "\nExecuting Cycle: " << std::dec << (i + 1) << std::endl;
        
        // This is your hardware clock tick!
        cpu.fetch();
        cpu.decode();
        cpu.execute();
        cpu.read();
        cpu.writeback();
    }

    // 6. Print final state to verify register values
    std::cout << "\n--- Final State ---" << std::endl;
    cpu.print_state();

    return 0;
}