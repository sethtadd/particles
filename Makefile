# ------------- #
# C++ VARIABLES #
# ------------- #

CXX = g++  # C++ Compiler
CXX_FLAGS = -Iinclude -Iinclude/glad -Wall  # Compiler flags
LD_FLAGS = -lglfw -lGL -lfftw3f -lsndfile -lportaudio -lcudart -L/usr/local/cuda/lib64  # Linker flags and required libraries

SRC_DIR=src
OBJ_DIR=build
BIN_DIR=bin

TARGET = $(BIN_DIR)/particles  # Binary output location
CXX_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CXX_OBJS = $(CXX_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# -------------- #
# CUDA VARIABLES #
# -------------- #

NVCC = nvcc  # CUDA Compiler
NVCC_FLAGS = -Iinclude -Iinclude/glad -diag-suppress 20012  # CUDA Compiler flags

CU_OBJS = build/ParticleSystem.cu.o build/CudaHelpers.cu.o  # CUDA Object files
CU_DEVICE_OBJ = build/device_link.cu.o # CUDA Object file for device linking

### ------------- ###
### LINKING RULES ###
### ------------- ###

# Linking
$(TARGET): build/gl.o $(CXX_OBJS) $(CU_OBJS) $(CU_DEVICE_OBJ)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LD_FLAGS)

# Device linking
$(CU_DEVICE_OBJ): $(CU_OBJS)
	$(NVCC) $(NVCC_FLAGS) --device-link $^ -o $@

### ----------------- ###
### COMPILE CPP FILES ###
### ----------------- ###

# Pattern rule to compile each source + header file pair
build/%.o: src/%.cpp include/%.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Pattern rule to compile source files (without a matching header file)
build/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

### ------------ ###
### COMPILE GLAD ###
### ------------ ###

build/gl.o: src/glad/gl.c include/glad/gl.h
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

### ---------------- ###
### COMPILE CU FILES ###
### ---------------- ###

build/ParticleSystem.cu.o: src/ParticleSystem.cu include/ParticleSystem.hpp
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) --device-c -c $< -o $@

build/CudaHelpers.cu.o: src/CudaHelpers.cu include/CudaHelpers.cuh
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) --device-c -c $< -o $@

### ----- ###
### OTHER ###
### ----- ###

# Phony targets
.PHONY: all run clean

# Default target
all: $(TARGET)

# Run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
