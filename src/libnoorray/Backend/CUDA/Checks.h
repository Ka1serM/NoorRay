#pragma once

#include <stdexcept>
#include <string>

#include "Backend/CUDA/Annotations.h"

#if defined(NR_CUDA_ACTIVE)
#include <cuda_runtime.h>
#include <optix.h>

namespace nr::cuda::detail {

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

}

#define NR_GPU_CHECK(call) ::nr::cuda::detail::checkCuda((call), #call, __FILE__, __LINE__)
#define NR_OPTIX_CHECK(call) ::nr::cuda::detail::checkOptix((call), #call, __FILE__, __LINE__)
#else
#define NR_GPU_CHECK(call) (call)
#define NR_OPTIX_CHECK(call) (call)
#endif
