#include "NoorRay/Render/Targets.h"

#include "NoorRay/Gpu/BufferUpload.h"
#include "NoorRay/Shaders/Denoise/DenoiseRecords.h"

namespace NoorRay::Render {

std::expected<void, Core::Error> Targets::open(NoorRHI::Device& device, const std::uint32_t filmWidth, const std::uint32_t filmHeight) {
    if (filmWidth == 0 || filmHeight == 0)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the film needs at least one pixel");
    const std::size_t count = std::size_t{filmWidth} * filmHeight;
    if (auto opened = Gpu::openZeroed(device, frame, count * 5); !opened)
        return opened;
    for (auto* buffer : {&accumulated, &output, &albedo, &normalDepth})
        if (auto opened = Gpu::openZeroed(device, *buffer, count); !opened)
            return opened;
    if (auto opened = Gpu::openZeroed(device, objectIds, count); !opened)
        return opened;
    if (auto opened = Gpu::openZeroed(device, pixels, count); !opened)
        return opened;
    lighting = {};
    demodulation = {};
    filmSums = {};
    width = filmWidth;
    height = filmHeight;
    return {};
}

std::expected<void, Core::Error> Targets::openLightingSplit(NoorRHI::Device& device) {
    if (lighting)
        return {};
    const std::size_t count = std::size_t{width} * height;
    if (auto opened = Gpu::openZeroed(device, lighting, count * Shaders::lightingStride); !opened)
        return opened;
    return Gpu::openZeroed(device, demodulation, count * 2);
}

std::expected<void, Core::Error> Targets::openFilmSums(NoorRHI::Device& device) {
    if (auto opened = openLightingSplit(device); !opened)
        return opened;
    if (filmSums)
        return {};
    //! The first frame of each accumulation overwrites the sums, so they need no zeros.
    return Core::check(filmSums.open(device, std::size_t{width} * height * Shaders::denoiseFilmStride));
}

Shaders::FilmRecord Targets::getRecord(const bool accumulate, const bool splitsLighting, const bool approximatesSpecular, const float exposure) const {
    const bool splits = splitsLighting && lighting;
    return {
        .frame = frame.getPtr(),
        .accumulated = accumulated.getPtr(),
        .output = output.getPtr(),
        .albedo = albedo.getPtr(),
        .normalDepth = normalDepth.getPtr(),
        .objectIds = objectIds.getPtr(),
        .pixels = pixels.getPtr(),
        .lighting = splits ? lighting.getPtr() : NoorRHI::AtomicPtr<float>{},
        .demodulation = splits ? demodulation.getPtr() : NoorRHI::Ptr<NoorRHI::float4>{},
        .width = width,
        .height = height,
        .accumulate = accumulate ? 1u : 0u,
        .splitsLighting = splits ? 1u : 0u,
        .shared = shared,
        .approximatesSpecular = approximatesSpecular ? 1u : 0u,
        .exposure = exposure,
    };
}
}
