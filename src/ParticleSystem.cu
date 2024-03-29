#include <vector>
#include <iostream>

#include <glad/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Shader.hpp"
#include "ParticleSystem.hpp"
#include "CudaHelpers.cuh"

__global__ void updateParticles(float3 *d_positions, float3 *d_velocities, float4 *d_colors, int numParticles, float deltaTime)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Using a loop here allows us to use a single thread to update multiple particles, reducing redundant thread launches
    for (int i = index; i < numParticles; i += stride)
    {
        if (i < numParticles)
        {
            // Apply gravity
            d_velocities[i] += deltaTime * make_float3(0.0f, -0.5f, 0.0f);
            // Update position
            d_positions[i] += deltaTime * d_velocities[i];

            // Collision detection
            float particleRadius = 0.01f;
            for (int j = 0; j < numParticles; ++j)
            {
                if (i != j)
                {
                    float3 r = d_positions[i] - d_positions[j];
                    float rNorm = norm(r);
                    if (rNorm < particleRadius)
                    {
                        // Collision
                        float3 v1 = d_velocities[i];
                        float3 v2 = d_velocities[j];
                        float3 v1New = v1 - 2 * v1 * r / (rNorm * rNorm) * r;
                        float3 v2New = v2 - 2 * v2 * r / (rNorm * rNorm) * r;
                        d_velocities[i] = v1New;
                        d_velocities[j] = v2New;
                        d_positions[i] += 1.01f * (particleRadius - rNorm) * r / rNorm;
                        d_positions[j] -= 1.01f * (particleRadius - rNorm) * r / rNorm;
                    }
                }
            }

            // Bounce off walls
            float bounceFactor = 0.8f;
            if (d_positions[i].x > 1.0f)
            {
                d_positions[i].x = 1.0f;
                d_velocities[i].x *= -bounceFactor;
            }
            else if (d_positions[i].x < -1.0f)
            {
                d_positions[i].x = -1.0f;
                d_velocities[i].x *= -bounceFactor;
            }

            if (d_positions[i].y > 1.0f)
            {
                d_positions[i].y = 1.0f;
                d_velocities[i].y *= -bounceFactor;
            }
            else if (d_positions[i].y < -1.0f)
            {
                d_positions[i].y = -1.0f;
                d_velocities[i].y *= -bounceFactor;
            }

            if (d_positions[i].z > 1.0f)
            {
                d_positions[i].z = 1.0f;
                d_velocities[i].z *= -bounceFactor;
            }
            else if (d_positions[i].z < -1.0f)
            {
                d_positions[i].z = -1.0f;
                d_velocities[i].z *= -bounceFactor;
            }
        }
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

void ParticleSystem::init(int numParticles, float particleRadius)
{
    numParticles_ = numParticles;
    particleRadius_ = particleRadius;

    bool is3D = false;

    particleShader_.init("shaders/particles.vertex.glsl", "shaders/particles.geometry.glsl", "shaders/particles.fragment.glsl");

    // Particle positions
    for (int i = 0; i < numParticles_; i++)
    {
        float3 position = make_float3(
            2.0f * (float)rand() / RAND_MAX - 1.0f,         // x
            2.0f * (float)rand() / RAND_MAX - 1.0f,         // y
            is3D * 2.0f * (float)rand() / RAND_MAX - 1.0f); // z
        h_positions_.push_back(position);
    }

    // Particle velocities
    float maxVelocity = 0.05f;
    for (int i = 0; i < numParticles_; i++)
    {
        float3 velocity = make_float3(
            (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity,         // x
            (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity,         // y
            is3D * (2.0f * rand() / RAND_MAX - 1.0f) * maxVelocity); // z
        h_velocities_.push_back(velocity);
    }

    // Particle colors
    for (int i = 0; i < numParticles_; i++)
    {
        float3 a = h_velocities_[i];
        float aNorm = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
        float v = aNorm / maxVelocity;
        v *= v; // Square velocity so color is proportional to kinetic energy
        float4 color = make_float4(
            v,        // r
            1.0f,     // g
            1.0f - v, // b
            1.0f);    // a
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
