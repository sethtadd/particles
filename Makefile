# Variables
CXX = g++                                          # Compiler
CXXFLAGS = -Iinclude -Wall                         # Compiler flags
LDFLAGS = -lglfw -lGL                              # Linker flags, add required libraries here
BIN = bin/particles                                # Binary output location
OBJ = build/particles.o build/Shader.o build/gl.o  # Object files

# Phony targets
.PHONY: all run clean

# Default target
all: $(BIN)

# Linking
$(BIN): $(OBJ)
	$(CXX) $(OBJ) -o $(BIN) $(LDFLAGS)

# Compilation
build/particles.o: src/particles.cpp
	$(CXX) $(CXXFLAGS) -c src/particles.cpp -o build/particles.o

build/Shader.o: src/Shader.cpp include/Shader.hpp
	$(CXX) $(CXXFLAGS) -c src/Shader.cpp -o build/Shader.o

build/gl.o: include/glad/gl.c include/glad/gl.h
	$(CXX) $(CXXFLAGS) -c include/glad/gl.c -o build/gl.o

# Run
run: $(BIN)
	./$(BIN)

# Clean
clean:
	rm -f build/*.o $(BIN)
