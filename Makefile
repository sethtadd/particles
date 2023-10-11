# Variables
CXX = g++  # C++ Compiler
NVCC = nvcc  # CUDA Compiler
CXXFLAGS = -Iinclude -Wall  # C++ Compiler flags
NVCCFLAGS = -Iinclude  # CUDA Compiler flags
LDFLAGS = -lglfw -lGL -lcudart -L/usr/local/cuda/lib64  # Linker flags and required libraries
BIN = bin/particles  # Binary output location
CXX_OBJ = build/particles.cu.o build/Shader.o build/gl.o  # C++ Object files
CU_OBJ = build/CudaKernels.cu.o  # CUDA Object files

# Phony targets
.PHONY: all run clean

# Default target
all: $(BIN)

# Linking
$(BIN): $(CXX_OBJ) $(CU_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compilation
build/particles.cu.o: src/particles.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o build/particles.cu.o

build/Shader.o: src/Shader.cpp include/Shader.hpp
	$(CXX) $(CXXFLAGS) -c src/Shader.cpp -o build/Shader.o

build/gl.o: include/glad/gl.c include/glad/gl.h
	$(CXX) $(CXXFLAGS) -c include/glad/gl.c -o build/gl.o

build/CudaKernels.cu.o: src/CudaKernels.cu
	$(NVCC) $(NVCCFLAGS) -c src/CudaKernels.cu -o build/CudaKernels.cu.o

# Run
run: $(BIN)
	./$(BIN)

# Clean
clean:
	rm -f build/*.o build/*.cu.o $(BIN)
