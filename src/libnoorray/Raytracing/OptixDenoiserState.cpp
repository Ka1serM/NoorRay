#include "Raytracing/OptixDenoiserState.h"

#include <algorithm>

#include <glm/vec4.hpp>
#include <optix_stubs.h>

#include "CUDA/Checks.h"

OptixDenoiserState::~OptixDenoiserState() noexcept
{
    reset();
}

void OptixDenoiserState::ensure(const OptixDeviceContext context,
    const cudaStream_t stream, const uint32_t width, const uint32_t height)
{
    if (denoiser == nullptr)
    {
        OptixDenoiserOptions options{};
        options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        NR_OPTIX_CHECK(optixDenoiserCreate(
            context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser));
    }
    if (configuredWidth == width && configuredHeight == height)
        return;

    OptixDenoiserSizes sizes{};
    NR_OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, width, height, &sizes));
    stateSize = sizes.stateSizeInBytes;
    scratchSize = std::max(
        sizes.withoutOverlapScratchSizeInBytes, sizes.computeIntensitySizeInBytes);
    state.allocate(stateSize, stream);
    scratch.allocate(scratchSize, stream);
    output.allocate(sizeof(glm::vec4) * static_cast<size_t>(width) * height, stream);
    intensity.allocate(sizeof(float), stream);
    NR_OPTIX_CHECK(optixDenoiserSetup(denoiser, stream, width, height,
        state.devicePtr(), stateSize, scratch.devicePtr(), scratchSize));
    configuredWidth = width;
    configuredHeight = height;
}

const void* OptixDenoiserState::run(const OptixDeviceContext context,
    const cudaStream_t stream, const void* input, const uint32_t width,
    const uint32_t height)
{
    ensure(context, stream, width, height);
    const size_t rowStride = static_cast<size_t>(width) * sizeof(glm::vec4);
    OptixDenoiserLayer layer{};
    layer.input = {reinterpret_cast<CUdeviceptr>(input), width, height,
        static_cast<unsigned int>(rowStride), sizeof(glm::vec4), OPTIX_PIXEL_FORMAT_FLOAT4};
    layer.output = {output.devicePtr(), width, height,
        static_cast<unsigned int>(rowStride), sizeof(glm::vec4), OPTIX_PIXEL_FORMAT_FLOAT4};

    NR_OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser, stream, &layer.input,
        intensity.devicePtr(), scratch.devicePtr(), scratchSize));
    OptixDenoiserParams params{};
    params.hdrIntensity = intensity.devicePtr();
    OptixDenoiserGuideLayer guide{};
    NR_OPTIX_CHECK(optixDenoiserInvoke(denoiser, stream, &params,
        state.devicePtr(), stateSize, &guide, &layer, 1, 0, 0,
        scratch.devicePtr(), scratchSize));
    return output.get();
}

void OptixDenoiserState::reset() noexcept
{
    output.reset();
    intensity.reset();
    state.reset();
    scratch.reset();
    if (denoiser != nullptr)
        optixDenoiserDestroy(denoiser);
    denoiser = nullptr;
    configuredWidth = 0;
    configuredHeight = 0;
    stateSize = 0;
    scratchSize = 0;
}
