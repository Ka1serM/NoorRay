#pragma once

#include <stdexcept>
#include <string>

#include "GPU/Annotations.h"

#if defined(NR_CUDA_ACTIVE)
#include <cuda_runtime.h>
#include <optix.h>

inline void checkCuda(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
            cudaGetErrorString(result));
    }
}

inline void checkOptix(OptixResult result, const char* expression, const char* file, int line)
{
    if (result != OPTIX_SUCCESS)
    {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
            optixGetErrorName(result));
    }
}

#define NR_GPU_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define NR_OPTIX_CHECK(call) checkOptix((call), #call, __FILE__, __LINE__)
#else
#define NR_GPU_CHECK(call) (call)
#define NR_OPTIX_CHECK(call) (call)
#endif
