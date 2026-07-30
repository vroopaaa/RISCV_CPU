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
	# Build Rules
	# ==========================================

# 'all' is the default target when you just type 'make'
all: $(TARGET)

# Rule to link all object files together into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build successful! Run with: ./$(TARGET)"

# Rule to compile each .cpp file into a .o file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to clean up compiled files (run 'make clean')
clean:
	rm -rf $(BUILD_DIR)/*
	@echo "Build directory cleaned!"

# Phony targets aren't actual files
.PHONY: all clean