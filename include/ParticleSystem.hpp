#ifndef PARTICLE_SYSTEM_HPP
#define PARTICLE_SYSTEM_HPP

#include <vector>

#include <glad/gl.h>

#include "Camera.hpp"
#include "Shader.hpp"

// Forward declarations of CUDA types to avoid including CUDA headers
// This way we can include this header in pure C++ files
struct float3;
struct float4;
struct cudaGraphicsResource;

class ParticleSystem
{
public:
    ParticleSystem();
    ~ParticleSystem();

    void init(int numParticles, float particleRadius);
    void update(float deltaTime);
    void render(Camera &camera);

private:
    int numParticles_;
    float particleRadius_;
    // h_ for host memory
    std::vector<float3> h_positions_;
    std::vector<float> h_ages_;
    std::vector<float4> h_colors_;
    // d_ for device memory
    float3 *d_positions_;
    float *d_ages_;
    float4 *d_colors_;
    // OpenGL resources
    Shader particleShader_;
    GLuint particleVao_;
    GLuint instancePositionsVbo_;
    GLuint instanceAgesVbo_;
    GLuint instanceColorsVbo_;
    // cuda graphics resources
    cudaGraphicsResource *cuda_positions_vbo_resource_;
    cudaGraphicsResource *cuda_ages_vbo_resource_;
    cudaGraphicsResource *cuda_colors_vbo_resource_;
};

#endif // PARTICLE_SYSTEM_HPP
