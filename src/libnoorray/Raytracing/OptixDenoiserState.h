#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>
#include <optix.h>

#include "CUDA/Unique/AsyncDeviceBuffer.h"

class OptixDenoiserState
{
public:
    OptixDenoiserState() = default;
    ~OptixDenoiserState() noexcept;

    OptixDenoiserState(const OptixDenoiserState&) = delete;
    OptixDenoiserState& operator=(const OptixDenoiserState&) = delete;

    const void* run(OptixDeviceContext context, cudaStream_t stream,
        const void* input, uint32_t width, uint32_t height);
    void reset() noexcept;

private:
    OptixDenoiser denoiser{};
    uint32_t configuredWidth{};
    uint32_t configuredHeight{};
    size_t stateSize{};
    size_t scratchSize{};
    nr::cuda::UniqueAsyncDeviceBuffer state;
    nr::cuda::UniqueAsyncDeviceBuffer scratch;
    nr::cuda::UniqueAsyncDeviceBuffer output;
    nr::cuda::UniqueAsyncDeviceBuffer intensity;

    void ensure(OptixDeviceContext context, cudaStream_t stream,
        uint32_t width, uint32_t height);
};
