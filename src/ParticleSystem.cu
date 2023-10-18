#include <vector>
#include <iostream>

#include <glad/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "ParticleSystem.hpp"
#include "CudaHelpers.cuh"

__global__ void updateParticles(float3 *d_positions, float3 *d_velocities, float4 *d_colors, int numParticles, float deltaTime)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        // Update position based on velocity
        d_positions[i].x += deltaTime * d_velocities[i].x;
        d_positions[i].y += deltaTime * d_velocities[i].y;
        d_positions[i].z += deltaTime * d_velocities[i].z;

        // Bounce off walls
        if (d_positions[i].x > 1.0f)
        {
            d_positions[i].x = 1.0f;
            d_velocities[i].x *= -1.0f;
        }
        else if (d_positions[i].x < -1.0f)
        {
            d_positions[i].x = -1.0f;
            d_velocities[i].x *= -1.0f;
        }

        if (d_positions[i].y > 1.0f)
        {
            d_positions[i].y = 1.0f;
            d_velocities[i].y *= -1.0f;
        }
        else if (d_positions[i].y < -1.0f)
        {
            d_positions[i].y = -1.0f;
            d_velocities[i].y *= -1.0f;
        }

        if (d_positions[i].z > 1.0f)
        {
            d_positions[i].z = 1.0f;
            d_velocities[i].z *= -1.0f;
        }
        else if (d_positions[i].z < -1.0f)
        {
            d_positions[i].z = -1.0f;
            d_velocities[i].z *= -1.0f;
        }

        // Get min/max velocities
        float maxVelocity = 0.0f;
        float minVelocity = 0.0f;
        for (int j = 0; j < numParticles; ++j)
        {
            float velocity = norm(d_velocities[j]);
            if (velocity > maxVelocity)
                maxVelocity = velocity;
            else if (velocity < minVelocity)
                minVelocity = velocity;
        }

        // Normalize velocity to [0, 1]
        float v = (norm(d_velocities[i]) - minVelocity) / (maxVelocity - minVelocity);
        v = v * v; // Square velocity so color is proportional to kinetic energy

        d_colors[i] = make_float4(
            v,        // r
            0.5f,     // g
            1.0f - v, // b
            1.0f);    // a
    }
}

ParticleSystem::ParticleSystem() {}

ParticleSystem::~ParticleSystem()
{
    // Clean up
    cudaGraphicsUnregisterResource(cuda_positions_vbo_resource_);
    cudaGraphicsUnregisterResource(cuda_velocities_vbo_resource_);
    cudaGraphicsUnregisterResource(cuda_colors_vbo_resource_);
    glDeleteVertexArrays(1, &particleVao_);
    glDeleteBuffers(1, &instancePositionsVbo_);
    glDeleteBuffers(1, &instanceVelocitiesVbo_);
}

void ParticleSystem::init(int numParticles)
{
    numParticles_ = numParticles;

    // Particle positions
    for (int i = 0; i < numParticles_; i++)
    {
        float3 position = make_float3(
            2.0f * (float)rand() / RAND_MAX - 1.0f,  // x
            2.0f * (float)rand() / RAND_MAX - 1.0f,  // y
            2.0f * (float)rand() / RAND_MAX - 1.0f); // z
        h_positions_.push_back(position);
    }

    // Particle velocities
    float maxVelocity = 0.05f;
    for (int i = 0; i < numParticles_; i++)
    {
        float3 velocity = make_float3(
            (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity,  // x
            (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity,  // y
            (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity); // z
        h_velocities_.push_back(velocity);
    }

    // Particle colors
    for (int i = 0; i < numParticles_; i++)
    {
        float4 color = make_float4(
            1.0f,  // r
            1.0f,  // g
            1.0f,  // b
            1.0f); // a
        h_colors_.push_back(color);
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
        GL_STATIC_DRAW);                // usage

    // Position attribute
    glVertexAttribPointer(
        0,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 1); // Update attribute every 1 instance

    // Instance velocities
    glGenBuffers(1, &instanceVelocitiesVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVelocitiesVbo_);
    glBufferData(
        GL_ARRAY_BUFFER,                // target
        numParticles_ * sizeof(float3), // size
        h_velocities_.data(),           // data
        GL_STATIC_DRAW);                // usage

    // Velocity attribute
    glVertexAttribPointer(
        1,                 // index
        3,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        3 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1); // Update attribute every 1 instance

    // Instance colors
    glGenBuffers(1, &instanceColorsVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceColorsVbo_);
    glBufferData(
        GL_ARRAY_BUFFER,                // target
        numParticles_ * sizeof(float4), // size
        h_colors_.data(),               // data
        GL_STATIC_DRAW);                // usage

    // Color attribute
    glVertexAttribPointer(
        2,                 // index
        4,                 // size
        GL_FLOAT,          // type
        GL_FALSE,          // normalized
        4 * sizeof(float), // stride
        (void *)0);        // pointer
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1); // Update attribute every 1 instance

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
    glBindVertexArray(0);             // Unbind VAO

    // Register instance VBOs with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_positions_vbo_resource_, instancePositionsVbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_velocities_vbo_resource_, instanceVelocitiesVbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_colors_vbo_resource_, instanceColorsVbo_, cudaGraphicsMapFlagsWriteDiscard);
}

void ParticleSystem::update(float deltaTime)
{
    // Map VBOs
    cudaGraphicsMapResources(1, &cuda_positions_vbo_resource_, 0);
    cudaGraphicsMapResources(1, &cuda_velocities_vbo_resource_, 0);
    cudaGraphicsMapResources(1, &cuda_colors_vbo_resource_, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions_, nullptr, cuda_positions_vbo_resource_);
    cudaGraphicsResourceGetMappedPointer((void **)&d_velocities_, nullptr, cuda_velocities_vbo_resource_);
    cudaGraphicsResourceGetMappedPointer((void **)&d_colors_, nullptr, cuda_colors_vbo_resource_);

    // Update particles
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles_ + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions_, d_velocities_, d_colors_, numParticles_, deltaTime);
    cudaError_t err_ = cudaDeviceSynchronize(); // Blocks execution until kernel is finished
    if (err_ != cudaSuccess)
    {
        std::cerr << "Failed to synchronize on the CUDA device (error code " << cudaGetErrorString(err_) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Unmap VBOs (release them for OpenGL)
    cudaGraphicsUnmapResources(1, &cuda_positions_vbo_resource_, 0);
    cudaGraphicsUnmapResources(1, &cuda_velocities_vbo_resource_, 0);
    cudaGraphicsUnmapResources(1, &cuda_colors_vbo_resource_, 0);
}

void ParticleSystem::render()
{
    glBindVertexArray(particleVao_);
    glDrawArraysInstanced(GL_POINTS, 0, 1, numParticles_);
    glBindVertexArray(0);
}
