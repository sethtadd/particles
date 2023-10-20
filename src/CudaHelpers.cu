#include <stdio.h>
#include "CudaHelpers.cuh"

__device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(float b, float3 a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 &operator+=(float3 &a, const float3 &b)
{
    a = a + b;
    return a;
}

__device__ float3 &operator-=(float3 &a, const float3 &b)
{
    a = a - b;
    return a;
}

__device__ float3 &operator*=(float3 &a, const float &b)
{
    a = a * b;
    return a;
}

__device__ float norm(float3 a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ float3 normalize(float3 a)
{
    return a / norm(a);
}

__device__ float operator*(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float smoothstep(float a, float b, float x)
{
    float t = (x - a) / (b - a);
    t = fmaxf(0.0f, fminf(1.0f, t));
    return t * t * (3.0f - 2.0f * t);
}