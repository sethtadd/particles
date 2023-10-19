#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

__device__ float3 operator*(float3 a, float b);
__device__ float3 operator*(float b, float3 a);

__device__ float3 operator/(float3 a, float b);

__device__ float3 operator+(float3 a, float3 b);

__device__ float3 operator-(float3 a, float3 b);

__device__ float3 &operator+=(float3 &a, const float3 &b);

__device__ float3 &operator-=(float3 &a, const float3 &b);

__device__ float norm(float3 a);

__device__ float3 normalize(float3 a);

__device__ float operator*(float3 a, float3 b);

__device__ float smoothstep(float a, float b, float x);

#endif // CUDA_HELPERS_CUH
