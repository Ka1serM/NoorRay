#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace nr::cuda::detail
{

inline void check(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression + ": " +
            cudaGetErrorString(result));
    }
}

}

#define NR_CUDA_UNIQUE_CHECK(call) ::nr::cuda::detail::check((call), #call, __FILE__, __LINE__)
