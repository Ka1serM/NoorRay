#pragma once

#include <cuda_runtime.h>
#include <glm/vec4.hpp>

// LUT texture handles — used by OpenPBR macros on the GPU.
#if defined(__CUDACC__)
extern __device__ cudaTextureObject_t openpbrLuts[8];
#endif

// Sampling helpers — device path actually reads textures; host path is a stub
// (these functions are never called from host code; nvcc's host pass needs them).
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ __forceinline__ glm::vec4 sample2D(cudaTextureObject_t tex, float u, float v)
{
    float4 v4 = tex2D<float4>(tex, u, v);
    return glm::vec4(v4.x, v4.y, v4.z, v4.w);
}
__device__ __forceinline__ glm::vec4 sample3D(cudaTextureObject_t tex, float u, float v, float w)
{
    float4 v4 = tex3D<float4>(tex, u, v, w);
    return glm::vec4(v4.x, v4.y, v4.z, v4.w);
}
#elif defined(__CUDACC__)
inline glm::vec4 sample2D(cudaTextureObject_t, float, float) { return glm::vec4(0.0f); }
inline glm::vec4 sample3D(cudaTextureObject_t, float, float, float) { return glm::vec4(0.0f); }
#endif

// OpenPBR texture LUT macros — device path accesses real textures,
// host path returns zeros (called functions are nvcc-host-compiled but never used).
#define OPENPBR_USE_TEXTURE_LUTS 1
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define OPENPBR_SAMPLE_2D_TEXTURE(id, uv) sample2D(openpbrLuts[id], (uv).x, (uv).y)
#define OPENPBR_SAMPLE_3D_TEXTURE(id, uvw) sample3D(openpbrLuts[id], (uvw).x, (uvw).y, (uvw).z)
#else
#define OPENPBR_SAMPLE_2D_TEXTURE(id, uv) glm::vec4(0.0f)
#define OPENPBR_SAMPLE_3D_TEXTURE(id, uvw) glm::vec4(0.0f)
#endif

// Host-side upload function (called once at startup).
void uploadOpenPbrLuts();
void destroyOpenPbrLuts();
