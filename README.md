# Chaotic Attractor Music Visualizer

This README was updated on (MM/DD/YYY): 10/23/2023

## Description

The purpose of this project is to recreate [this music visualizer (CONTAINS FLASHING LIGHTS)](https://youtu.be/G6m-d52-HP8?si=lthNN81XR5K6B-id) that I made in 2017/2018, but from "scratch" and with much higher performance. The original visualizer was made in [Processing 3](https://processing.org/).

## TODO

### Core

- [x] ~~OpenGL particles~~
- [x] ~~CUDA computing~~
- [x] ~~Audio playback (and pipe audio data to CUDA kernel)~~
- [ ] FFT preprocessing kernel
- [ ] Audio visualization helpers (e.g. smoothing, beat detection?, etc.)

### Extras

- [ ] Post-processing effects (e.g. bloom/glare, chromatic aberration, depth of field, etc.)
- [ ] Physics loop independent of rendering loop
- [ ] Refactor/reduce Makefile
- [ ] UML diagram of class structure, UML diagram of CPU/GPU/IO data flow

## Screenshots

![A rendition of the Sprott attractor](./sprott.jpeg)
![A rendition of the three scroll attractor](./three_scroll.jpeg)
<!-- <img src="./sprott.jpeg" width="400" height="400" alt="A rendition of the Sprott attractor"> -->
<!-- <img src="./three_scroll.jpeg" width="400" height="400" alt="A rendition of the three scroll attractor"> -->

## Libraries/APIs Used (Dependencies)

- OpenGL - *graphics API*
- GLFW - *windowing library*
- CUDA - *parallel computing platform*
- PortAudio - *audio device I/O library*
- libsndfile - *audio file I/O library*

## How to Run

1. Make sure you have the dependencies installed (see above).
2. Place an audio file in `/audio`, update the name of the audio file being read in `Main.cpp` (currently `follow.wav`).
3. Then simply do

    ```bash
    make run
    ```

    to build and run the program.

## File Structure

Building involves the `g++` and `nvcc` compilers. The `g++` compiler compiles `.cpp` files into object files and `nvcc` compiles `.cu` files into object files. Then `g++` links all the object files into a single executable.

```bash
.
├── audio  # audio files ("follow.wav" not included in repo)
│   └── follow.wav
├── bin  # linked binary executable goes here
├── build  # compiled object files go here
├── include  # header files
│   ├── AudioPlayer.hpp
│   ├── Camera.hpp
│   ├── CudaHelpers.cuh
│   ├── CudaUnifiedMemory.hpp
│   ├── Framebuffer.hpp
│   ├── glad
│   │   └── gl.h
│   ├── ParticleSystem.hpp
│   └── Shader.hpp
├── Makefile
├── README.md
├── shaders  # OpenGL shader files
│   ├── hdr.fragment.glsl
│   ├── hdr.vertex.glsl
│   ├── particles.fragment.glsl
│   ├── particles.geometry.glsl
│   └── particles.vertex.glsl
└── src  # source files
    ├── AudioPlayer.cpp
    ├── Camera.cpp
    ├── CudaHelpers.cu
    ├── CudaUnifiedMemory.cu
    ├── Framebuffer.cpp
    ├── glad
    │   └── gl.c
    ├── Main.cpp
    ├── ParticleSystem.cu
    └── Shader.cpp
```
