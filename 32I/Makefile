# ==========================================
# RISC-V CPU Emulator Makefile
# ==========================================

# 1. Compiler and Flags
CXX = g++
# -Wall -Wextra: Enable warnings to catch bugs early
# -Iinclude: Tells the compiler to look in the 'include' folder for .h files
# -g: Adds debugging information (useful if you use GDB later)
# -std=c++11: Uses modern C++ standards
CXXFLAGS = -Wall -Wextra -Iinclude -g -std=c++11

# 2. Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# 3. Files
# Automatically find all .cpp files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Convert the list of .cpp files into a list of .o (object) files in the build dir
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# 4. Target Executable Name
TARGET = $(BUILD_DIR)/cpu_emulator

# ==========================================
# Build Rules (main simulator)
# ==========================================

# 'all' is the default target when you just type 'make'
all: $(TARGET)

# Rule to link all object files together into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build successful! Run with: ./$(TARGET)"

# Rule to compile each .cpp file into a .o file. Every object depends on every
# header: a class layout change (e.g. a new CPU member) must rebuild all users,
# otherwise old objects disagree about sizeof/offsets and crash at run time.
HEADERS = $(wildcard $(INC_DIR)/*.h)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to clean up compiled files (run 'make clean')
clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(TEST_DIR)/test.elf $(TEST_DIR)/test.bin $(TEST_DIR)/test.dis
	@echo "Build directory cleaned!"

# ==========================================
# RISC-V Test Program Build (separate toolchain)
# ==========================================
RISCV_CC = riscv64-unknown-elf-gcc
RISCV_OBJCOPY = riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = riscv64-unknown-elf-objdump
RISCV_FLAGS = -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles -ffreestanding -O0

TEST_DIR = tests/basic

# Compiles start.s + main.c into test.elf, then extracts test.bin
$(TEST_DIR)/test.bin: $(TEST_DIR)/start.s $(TEST_DIR)/main.c $(TEST_DIR)/link.ld
	$(RISCV_CC) $(RISCV_FLAGS) -T $(TEST_DIR)/link.ld \
	  -o $(TEST_DIR)/test.elf $(TEST_DIR)/start.s $(TEST_DIR)/main.c
	$(RISCV_OBJCOPY) -O binary $(TEST_DIR)/test.elf $(TEST_DIR)/test.bin
	$(RISCV_OBJDUMP) -d $(TEST_DIR)/test.elf > $(TEST_DIR)/test.dis
	@echo "Test binary built: $(TEST_DIR)/test.bin (see test.dis for disassembly)"

test: $(TEST_DIR)/test.bin
	@echo "Test binary ready. Run 'make run-test' to execute it on your simulator."

# ==========================================
# Test Harness Build (runs test.bin on your CPU/Memory model)
# ==========================================
TEST_TARGET = $(BUILD_DIR)/run_test

# Linked directly against CPU.cpp/memory.cpp/NPU.cpp (+ SIMT.cpp/GridLauncher.cpp:
# CPU.cpp's LAUNCH opcode calls into the GPU) -- NOT src/main.cpp,
# since that already has its own main() and would clash with tests/basic/main.cpp
$(TEST_TARGET): $(TEST_DIR)/main.cpp $(TEST_DIR)/loader.cpp $(SRC_DIR)/CPU.cpp $(SRC_DIR)/memory.cpp $(SRC_DIR)/NPU.cpp $(SRC_DIR)/NPU_print.cpp $(SRC_DIR)/SIMT.cpp $(SRC_DIR)/GridLauncher.cpp $(INC_DIR)/NPU.h $(INC_DIR)/CPU.h $(INC_DIR)/memory.h $(INC_DIR)/SIMT.h $(INC_DIR)/GridLauncher.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp,$^)
	@echo "Test harness built! Run with: ./$(TEST_TARGET)"

$(BUILD_DIR)/run_test_t%: $(TEST_DIR)/main.cpp $(TEST_DIR)/loader.cpp $(SRC_DIR)/CPU.cpp $(SRC_DIR)/memory.cpp $(SRC_DIR)/NPU.cpp $(SRC_DIR)/NPU_print.cpp $(SRC_DIR)/SIMT.cpp $(SRC_DIR)/GridLauncher.cpp $(INC_DIR)/NPU.h $(INC_DIR)/CPU.h $(INC_DIR)/memory.h $(INC_DIR)/SIMT.h $(INC_DIR)/GridLauncher.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DNPU_MAX_DIM=$* -o $@ $(filter %.cpp,$^)
	@echo "Tile-$* test harness built! Run with: ./$@"

run-test: $(TEST_DIR)/test.bin $(TEST_TARGET)
	./$(TEST_TARGET)

# ==========================================
# SIMT layer (see include/SIMT.h)
# ==========================================
# SIMT.cpp calls CPU::decode_one/alu_exec/branch_taken (friend access, see
# CPU.h), and Memory pulls in NPU -- so every SIMT test links the same four
# .cpp files as the main scalar test harness above, plus SIMT.cpp itself.
SIMT_SRCS = $(SRC_DIR)/SIMT.cpp $(SRC_DIR)/GridLauncher.cpp $(SRC_DIR)/CPU.cpp $(SRC_DIR)/memory.cpp $(SRC_DIR)/NPU.cpp $(SRC_DIR)/NPU_print.cpp
SIMT_DEPS = $(SIMT_SRCS) $(INC_DIR)/SIMT.h $(INC_DIR)/GridLauncher.h $(INC_DIR)/CPU.h $(INC_DIR)/memory.h $(INC_DIR)/NPU.h

SIMT_REGFILE_TEST = $(BUILD_DIR)/simt_regfile_test

$(SIMT_REGFILE_TEST): tests/simt/regfile_test.cpp $(SIMT_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ tests/simt/regfile_test.cpp $(SIMT_SRCS)

run-simt-regfile-test: $(SIMT_REGFILE_TEST)
	./$(SIMT_REGFILE_TEST)

# Execution-engine tests: real C kernels (tests/simt/c_tests/*.c) using the
# .insn-wrapped macros in tests/simt/simt_isa.h, cross-compiled by the
# unmodified riscv64-unknown-elf-gcc and run through tests/simt/harness.cpp
# -- same methodology as tests/npu/run_npu_tests.py (compile, run, dump
# memory, diff against an independently-computed expected result in
# Python), not a C++ unit test hand-assembling instruction words.
SIMT_HARNESS = $(BUILD_DIR)/simt_harness

$(SIMT_HARNESS): tests/simt/harness.cpp tests/basic/loader.cpp $(SIMT_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ tests/simt/harness.cpp tests/basic/loader.cpp $(SIMT_SRCS)

run-simt-exec-tests: $(SIMT_HARNESS)
	python3 tests/simt/run_simt_tests.py

# Grid-launch kernel tests: hand-written assembly kernels (tests/simt/grid_tests/*.s)
# launched through GridLauncher by tests/simt/grid_harness.cpp, diffed in Python.
GRID_HARNESS = $(BUILD_DIR)/grid_harness

$(GRID_HARNESS): tests/simt/grid_harness.cpp tests/basic/loader.cpp $(SIMT_DEPS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ tests/simt/grid_harness.cpp tests/basic/loader.cpp $(SIMT_SRCS)

run-grid-tests: $(GRID_HARNESS) $(TEST_TARGET)
	python3 tests/simt/run_grid_tests.py

run-simt-tests: run-simt-regfile-test run-simt-exec-tests run-grid-tests

# Phony targets aren't actual files
.PHONY: all clean test run-test run-simt-regfile-test run-grid-tests run-simt-exec-tests run-simt-tests