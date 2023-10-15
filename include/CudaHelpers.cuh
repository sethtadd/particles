#ifndef CUDA_HELPERS_HPP
#define CUDA_HELPERS_HPP

__device__ float3 operator*(float3 a, float b);

__device__ float3 operator/(float3 a, float b);

__device__ float3 operator+(float3 a, float3 b);

__device__ float3 operator-(float3 a, float3 b);

__device__ float norm(float2 a);

__device__ float norm(float3 a);

__device__ float norm(float4 a);

__device__ float3 normalize(float3 a);

__device__ float operator*(float3 a, float3 b);

#endif // CUDA_HELPERS_HPP
