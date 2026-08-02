#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>
#include <optix.h>

#include "Backend/CUDA/Unique/AsyncDeviceBuffer.h"

namespace nr
{

class OptixDenoiser
{
public:
    OptixDenoiser() = default;
    ~OptixDenoiser() noexcept;

    OptixDenoiser(const OptixDenoiser&) = delete;
    OptixDenoiser& operator=(const OptixDenoiser&) = delete;

    const void* run(OptixDeviceContext context, cudaStream_t stream,
        const void* input, const void* albedoGuide, const void* normalGuide,
        void* output, uint32_t width, uint32_t height);
    void reset() noexcept;

private:
    ::OptixDenoiser denoiser{};
    uint32_t configuredWidth{};
    uint32_t configuredHeight{};
    size_t stateSize{};
    size_t scratchSize{};
    bool albedoGuideEnabled{};
    bool normalGuideEnabled{};
    nr::cuda::UniqueAsyncDeviceBuffer state;
    nr::cuda::UniqueAsyncDeviceBuffer scratch;
    nr::cuda::UniqueAsyncDeviceBuffer output;
    nr::cuda::UniqueAsyncDeviceBuffer intensity;

    void ensure(OptixDeviceContext context, cudaStream_t stream,
        uint32_t width, uint32_t height, bool useAlbedoGuide,
        bool useNormalGuide);
};

} // namespace nr
