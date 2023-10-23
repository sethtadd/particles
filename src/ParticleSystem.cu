#include <vector>
#include <iostream>
#include <random>

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

// Found these attractors at https://www.dynamicmath.xyz/strange-attractors/

__device__ float3 aizawaAttractor(float3 position)
{
    position *= 4.0f; // Scale down to make attractor more visible

    float a = 0.95f;
    float b = 0.7f;
    float c = 0.6f;
    float d = 3.5f;
    float e = 0.25f;
    float f = 0.1f;

    float x = position.x;
    float y = position.y;
    float z = position.z;

    float3 velocity = make_float3(
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - (z * z * z) / 3.0f - (x * x + y * y) * (1.0f + e * z) + f * z * x * x * x);

    return velocity;
}

__device__ float3 lorentzAttractor(float3 position)
{
    position.z += 0.3f;
    position *= 50.0f; // Scale down to make attractor more visible

    float sigma = 10.0f;
    float rho = 28.0f;
    float beta = 8.0f / 3.0f;

    float3 velocity = make_float3(
        sigma * (position.y - position.x),            // dx/dt
        position.x * (rho - position.z) - position.y, // dy/dt
        position.x * position.y - beta * position.z   // dz/dt
    );

    return velocity;
}

__device__ float3 halvorsenAttractor(float3 position)
{
    position *= 20.0f; // Scale down to make attractor more visible

    float a = 1.4f;

    float x = position.x;
    float y = position.y;
    float z = position.z;

    float3 velocity = make_float3(
        -a * x - 4.0f * y - 4.0f * z - y * y, // dx/dt
        -a * y - 4.0f * z - 4.0f * x - z * z, // dy/dt
        -a * z - 4.0f * x - 4.0f * y - x * x  // dz/dt
    );

    return velocity;
}

__device__ float3 rabinovichFabrikantAttractor(float3 position)
{
    position *= 5.0f; // Scale down to make attractor more visible

    float a = 0.14f;
    float b = 0.1f;

    float x = position.x;
    float y = position.y;
    float z = position.z;

    float3 velocity = make_float3(
        y * (z - 1.0f + x * x) + b * x,        // dx/dt
        x * (3.0f * z + 1.0f - x * x) + b * y, // dy/dt
        -2.0f * z * (a + x * y)                // dz/dt
    );

    return velocity;
}

__device__ float3 threeScrollAttractor(float3 position)
{
    position.z += 0.5f;
    position *= 400.0f; // Scale down to make attractor more visible

    float a = 32.48f;
    float b = 45.84f;
    float c = 1.18f;
    float d = 0.13f;
    float e = 0.57f;
    float f = 14.7f;

    float x = position.x;
    float y = position.y;
    float z = position.z;

    float3 velocity = make_float3(
        a * (y - x) + d * x * z,  // dx/dt
        b * x - x * z + f * y,    // dy/dt
        c * z + x * y - e * x * x // dz/dt
    );

    return velocity;
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

__global__ void updateParticles(float3 *d_positions, float *d_ages, float4 *d_colors, int numParticles, float deltaTime, float particleLifetime, int attractorIndex, float *audioData, int audioDataSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize random number generator
    unsigned long long int clockCount = clock64();
    curandState rngState;
    curand_init(clockCount, index, 0, &rngState);

    // Using a loop here allows us to use a single thread to update multiple particles, reducing redundant thread launches
    for (int i = index; i < numParticles; i += stride)
    {
        if (i < numParticles)
        {
            // Update age
            d_ages[i] += deltaTime;

            // Update position based on attractor
            float3 velocity;
            switch (attractorIndex)
            {
            case 0:
                velocity = sprottAttractor(d_positions[i]);
                break;
            case 1:
                velocity = halvorsenAttractor(d_positions[i]);
                break;
            case 2:
                velocity = aizawaAttractor(d_positions[i]);
                break;
            case 3:
                velocity = threeScrollAttractor(d_positions[i]);
                break;
            case 4:
            default: // added default case to handle unexpected index values
                velocity = lorentzAttractor(d_positions[i]);
                break;
            }

            velocity = normalize(velocity); // Quicker convergence to attractor
            d_positions[i] += deltaTime * velocity;

            // Respawn particles that die (age > maxAge) or travel too far from origin
            if (d_ages[i] > particleLifetime || norm(d_positions[i]) > 4.0f)
            {
                d_ages[i] = 0.0f;
                d_positions[i] = make_float3(
                    curand_normal(&rngState),
                    curand_normal(&rngState),
                    curand_normal(&rngState));
            }

            // Update color based on age
            float c = d_ages[i] / particleLifetime;
            c = sqrtf(c); // Emphasize younger particles
            d_colors[i] = make_float4(c, 0.5f, 1.0f - c, c);
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

    // Random number generator
    std::random_device rngDevice;
    std::mt19937 rng(rngDevice());
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
    std::normal_distribution<float> normalDist(0.0f, 1.0f);

    // Particle positions
    for (int i = 0; i < numParticles_; i++)
    {
        float3 position = make_float3(
            normalDist(rng), // x
            normalDist(rng), // y
            normalDist(rng)  // z
        );
        h_positions_.push_back(position);
    }

    // Particle ages
    particleLifetime_ = 20.0f;
    for (int i = 0; i < numParticles_; i++)
    {
        float age = uniformDist(rng) * particleLifetime_;
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
        GL_DYNAMIC_DRAW);               // usage

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
        GL_DYNAMIC_DRAW);              // usage

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
        GL_DYNAMIC_DRAW);               // usage

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

void ParticleSystem::update(float deltaTime, int attractorIndex, float *audioData, int audioDataSize)
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
    // TODO - FFT audio preprocessing BEFORE passing to updateParticles kernel, don't do FFT preprocessing in each updateParticles kernel thread
    // TODO - Audio should be passed to preprocessing kernel, FFT'd, then relevant values from FFT should be passed to updateParticles kernel
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions_,
        d_ages_,
        d_colors_,
        numParticles_,
        deltaTime,
        particleLifetime_,
        attractorIndex,
        audioData,
        audioDataSize);
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
