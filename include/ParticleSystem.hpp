#ifndef PARTICLE_SYSTEM_HPP
#define PARTICLE_SYSTEM_HPP

#include <vector>

#include <glad/gl.h>

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

    void init(int numParticles);
    void update();
    void render();

private:
    int numParticles_;
    // h_ for host memory
    std::vector<float3> h_positions_;
    std::vector<float3> h_velocities_;
    std::vector<float4> h_colors_;
    // d_ for device memory
    float3 *d_positions_;
    float3 *d_velocities_;
    float4 *d_colors_;
    // vbo variables
    GLuint particleVao_;
    GLuint instancePositionsVbo_;
    GLuint instanceVelocitiesVbo_;
    GLuint instanceColorsVbo_;
    // cuda graphics resources
    cudaGraphicsResource *cuda_positions_vbo_resource_;
    cudaGraphicsResource *cuda_velocities_vbo_resource_;
    cudaGraphicsResource *cuda_colors_vbo_resource_;
};

#endif // PARTICLE_SYSTEM_HPP
