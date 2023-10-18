#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH

#include <vector>

#include <glad/gl.h>

__global__ void updateParticles(float3 *d_positions, float3 *d_velocities, float4 *d_colors, int numParticles);

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

    cudaError_t err_;
};

#endif // PARTICLE_SYSTEM_CUH
