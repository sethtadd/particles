#include <vector>
#include <iostream>

#include <glad/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "Shader.hpp"
#include "ParticleSystem.hpp"
#include "CudaHelpers.cuh"

__device__ float3 velocityField(float3 position)
{
    float3 velocity = position;

    return velocity;
}

__device__ float3 lorentzAttractor(float3 position)
{
    position *= 50.0f; // Scale down to make attractor more visible
    position.z += 15.0f;

    float sigma = 10.0f;
    float rho = 28.0f;
    float beta = 8.0f / 3.0f;

    float3 velocity = make_float3(
        sigma * (position.y - position.x),            // dx/dt
        position.x * (rho - position.z) - position.y, // dy/dt
        position.x * position.y - beta * position.z   // dz/dt
    );

    return velocity / 100.0f; // Scale down to make attractor more visible
}

__device__ float3 sprottAttractor(float3 position)
{
    position *= 3.0f; // Scale down to make attractor more visible

    float a = 2.07f;
    float b = 1.79f;

    float x = position.x;
    float y = position.y;
    float z = position.z;

    float3 velocity = make_float3(
        y + a * x * y + x * z, // dx/dt
        1 - b * x * x + y * z, // dy/dt
        x - x * x - y * y      // dz/dt
    );

    return velocity;
}

__global__ void updateParticles(float3 *d_positions, float *d_ages, float4 *d_colors, int numParticles, float deltaTime, float particleLifetime)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize random number generator
    unsigned long long int clockCount = clock64();
    curandState state;
    curand_init(clockCount, index, 0, &state);

    // Using a loop here allows us to use a single thread to update multiple particles, reducing redundant thread launches
    for (int i = index; i < numParticles; i += stride)
    {
        if (i < numParticles)
        {
            // Update age
            d_ages[i] += deltaTime;

            // Update position based on attractor
            float3 velocity = sprottAttractor(d_positions[i]);
            velocity = normalize(velocity); // Quicker convergence to attractor
            d_positions[i] += deltaTime * velocity;

            // Respawn particles that die (age > maxAge)
            if (d_ages[i] > particleLifetime)
            {
                d_ages[i] = 0.0f;
                d_positions[i] = make_float3(
                    2.0f * curand_uniform(&state) - 1.0f,
                    2.0f * curand_uniform(&state) - 1.0f,
                    2.0f * curand_uniform(&state) - 1.0f);
            }

            // Respawn particles that go out of bounds
            float maxBound = 1.0f;
            if (abs(d_positions[i].x) > maxBound || abs(d_positions[i].y) > maxBound || abs(d_positions[i].z) > maxBound)
            {
                d_ages[i] = 0.0f;
                d_positions[i] = make_float3(
                    maxBound * (2.0f * curand_uniform(&state) - 1.0f),
                    maxBound * (2.0f * curand_uniform(&state) - 1.0f),
                    maxBound * (2.0f * curand_uniform(&state) - 1.0f));
            }

            // Update color based on age
            float c = d_ages[i] / particleLifetime;
            c = sqrtf(c); // Emphasize younger particles
            d_colors[i] = make_float4(c, 0.5f, 1.0f - c, sqrt(c));
        }
    }
}

ParticleSystem::ParticleSystem() {}

ParticleSystem::~ParticleSystem()
{
    // Clean up
    cudaGraphicsUnregisterResource(cuda_positions_vbo_resource_);
    cudaGraphicsUnregisterResource(cuda_ages_vbo_resource_);
    cudaGraphicsUnregisterResource(cuda_colors_vbo_resource_);
    glDeleteVertexArrays(1, &particleVao_);
    glDeleteBuffers(1, &instancePositionsVbo_);
    glDeleteBuffers(1, &instanceAgesVbo_);
}

void ParticleSystem::init(int numParticles, float particleRadius)
{
    numParticles_ = numParticles;
    particleRadius_ = particleRadius;

    particleShader_.init("shaders/particles.vertex.glsl", "shaders/particles.geometry.glsl", "shaders/particles.fragment.glsl");

    // Particle positions
    for (int i = 0; i < numParticles_; i++)
    {
        float3 position = make_float3(
            2.0f * (float)rand() / RAND_MAX - 1.0f,         // x
            2.0f * (float)rand() / RAND_MAX - 1.0f,         // y
            2.0f * (float)rand() / RAND_MAX - 1.0f); // z
        h_positions_.push_back(position);
    }

    // Particle ages
    float maxAge = 20.0f;
    for (int i = 0; i < numParticles_; i++)
    {
        float age = maxAge * (float)rand() / RAND_MAX;
        h_ages_.push_back(age);
    }

    // Particle colors
    for (int i = 0; i < numParticles_; i++)
    {
        float4 white = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        h_colors_.push_back(white);
    }

    // Particle VAO (for instanced rendering)
    glGenVertexArrays(1, &particleVao_);
    glBindVertexArray(particleVao_);

    // Instance positions
    glGenBuffers(1, &instancePositionsVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, instancePositionsVbo_);
    glBufferData(
        GL_ARRAY_BUFFER,                // target
        numParticles_ * sizeof(float3), // size
        h_positions_.data(),            // data
        GL_DYNAMIC_DRAW);                // usage

    // Position attribute
    glVertexAttribPointer(
        0,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // initial offset
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 1); // Update attribute every 1 instance

    // Instance ages
    glGenBuffers(1, &instanceAgesVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceAgesVbo_);
    glBufferData(
        GL_ARRAY_BUFFER,               // target
        numParticles_ * sizeof(float), // size
        h_ages_.data(),                // data
        GL_DYNAMIC_DRAW);               // usage

    // Age attribute
    glVertexAttribPointer(
        1,                 // index
        1,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        1 * sizeof(float), // stride
        (void *)0);        // initial offset
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1); // Update attribute every 1 instance

    // Instance colors
    glGenBuffers(1, &instanceColorsVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorsVbo_);
    glBufferData(
        GL_ARRAY_BUFFER,                // target
        numParticles_ * sizeof(float4), // size
        h_colors_.data(),               // data
        GL_DYNAMIC_DRAW);                // usage

    // Color attribute
    glVertexAttribPointer(
        2,                 // index
        4,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        4 * sizeof(float), // stride
        (void *)0);        // initial offset
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1); // Update attribute every 1 instance

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
    glBindVertexArray(0);             // Unbind VAO

    // Register instance VBOs with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_positions_vbo_resource_, instancePositionsVbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_ages_vbo_resource_, instanceAgesVbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_colors_vbo_resource_, instanceColorsVbo_, cudaGraphicsMapFlagsWriteDiscard);
}

void ParticleSystem::update(float deltaTime)
{
    // Map VBOs
    cudaGraphicsMapResources(1, &cuda_positions_vbo_resource_, 0);
    cudaGraphicsMapResources(1, &cuda_ages_vbo_resource_, 0);
    cudaGraphicsMapResources(1, &cuda_colors_vbo_resource_, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions_, nullptr, cuda_positions_vbo_resource_);
    cudaGraphicsResourceGetMappedPointer((void **)&d_ages_, nullptr, cuda_ages_vbo_resource_);
    cudaGraphicsResourceGetMappedPointer((void **)&d_colors_, nullptr, cuda_colors_vbo_resource_);

    // Update particles
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles_ + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions_, d_ages_, d_colors_, numParticles_, deltaTime, 20.0f);
    cudaError_t err_ = cudaDeviceSynchronize(); // Blocks execution until kernel is finished
    if (err_ != cudaSuccess)
    {
        std::cerr << "Failed to synchronize on the CUDA device (error code " << cudaGetErrorString(err_) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Unmap VBOs (release them for OpenGL)
    cudaGraphicsUnmapResources(1, &cuda_positions_vbo_resource_, 0);
    cudaGraphicsUnmapResources(1, &cuda_ages_vbo_resource_, 0);
    cudaGraphicsUnmapResources(1, &cuda_colors_vbo_resource_, 0);
}

void ParticleSystem::render(Camera &camera)
{
    particleShader_.use();
    particleShader_.setMatrix4f("view", camera.getViewMatrix());
    particleShader_.setMatrix4f("projection", camera.getProjectionMatrix());
    particleShader_.setFloat("particleRadius", particleRadius_);

    glBindVertexArray(particleVao_);
    glDepthMask(GL_FALSE);
    glDrawArraysInstanced(GL_POINTS, 0, 1, numParticles_);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
}
