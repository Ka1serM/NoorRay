#include "NoorRay/Render/Targets.h"

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Render {

std::expected<void, Core::Error> Targets::open(NoorRHI::Device& device, const std::uint32_t filmWidth, const std::uint32_t filmHeight) {
    if (filmWidth == 0 || filmHeight == 0)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the film needs at least one pixel");
    const std::size_t count = std::size_t{filmWidth} * filmHeight;
    if (auto opened = Gpu::openZeroed(device, frame, count * 4); !opened)
        return opened;
    for (auto* buffer : {&accumulated, &output, &albedo, &normalDepth})
        if (auto opened = Gpu::openZeroed(device, *buffer, count); !opened)
            return opened;
    if (auto opened = Gpu::openZeroed(device, objectIds, count); !opened)
        return opened;
    if (auto opened = Gpu::openZeroed(device, pixels, count); !opened)
        return opened;
    width = filmWidth;
    height = filmHeight;
    return {};
}

Shaders::FilmRecord Targets::getRecord(const bool accumulate) const {
    return {
        .frame = frame.getPtr(),
        .accumulated = accumulated.getPtr(),
        .output = output.getPtr(),
        .albedo = albedo.getPtr(),
        .normalDepth = normalDepth.getPtr(),
        .objectIds = objectIds.getPtr(),
        .pixels = pixels.getPtr(),
        .width = width,
        .height = height,
        .accumulate = accumulate ? 1u : 0u,
    };
}

}
