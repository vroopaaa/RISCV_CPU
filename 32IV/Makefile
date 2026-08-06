# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Target executable name
TARGET = $(BUILD_DIR)/emulator

# Find all .cpp files in src/
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Map the .cpp source files to .o object files in build/
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# Default target
all: $(TARGET)

# Link all object files to create the executable
$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build successful! Executable is located at $(TARGET)"

# Compile each .cpp file into a .o file
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf $(BUILD_DIR)/*
	@echo "Cleaned build directory."

# Phony targets to prevent conflicts with files named 'all' or 'clean'
.PHONY: all clean