# Variables
CXX = g++  # C++ Compiler
NVCC = nvcc  # CUDA Compiler
CXXFLAGS = -Iinclude -Iinclude/glad -Wall  # C++ Compiler flags
NVCCFLAGS = -Iinclude -Iinclude/glad -diag-suppress 20012  # CUDA Compiler flags
LDFLAGS = -lglfw -lGL -lcudart -L/usr/local/cuda/lib64  # Linker flags and required libraries
BIN = bin/particles  # Binary output location
CXX_OBJS = build/Shader.o build/Camera.o build/gl.o  # C++ Object files
CU_OBJS = build/particles.cu.o build/CudaHelpers.cu.o  # CUDA Object files
CU_DEVICE_OBJ = build/device_link.cu.o # CUDA Object file for device linking

# Phony targets
.PHONY: all run clean

# Default target
all: $(BIN)

# Linking
$(BIN): $(CXX_OBJS) $(CU_OBJS) $(CU_DEVICE_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Device linking
$(CU_DEVICE_OBJ): $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) --device-link $^ -o $@

# Compilation
build/particles.cu.o: src/particles.cu
	$(NVCC) $(NVCCFLAGS) --device-c -c $^ -o $@

build/Shader.o: src/Shader.cpp include/Shader.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/Camera.o: src/Camera.cpp include/Camera.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/gl.o: src/glad/gl.c include/glad/gl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/CudaHelpers.cu.o: src/CudaHelpers.cu include/CudaHelpers.cuh
	$(NVCC) $(NVCCFLAGS) --device-c -c $< -o $@

# Run
run: $(BIN)
	./$(BIN)

# Clean
clean:
	rm -f build/*.o build/*.cu.o bin/*
