#include "NoorRay/Render/Wavefront/PathPool.h"

namespace NoorRay::Render {

std::expected<void, Core::Error> PathPool::open(NoorRHI::Device& device, const std::uint32_t capacity) {
    if (capacity == 0)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the path pool needs at least one slot");
    const auto openAll = [&](auto&... buffers) -> std::expected<void, Core::Error> {
        std::expected<void, NoorRHI::Error> result;
        ((result = result ? buffers.open(device, capacity) : result), ...);
        return Core::check(std::move(result));
    };
    if (auto opened = openAll(origin, direction, throughput, radiance, state, hit, hitInstance, shadowOrigin, shadowDirection, shadowRadiance, pathBin,
                              closures, lightPath, minRayPdf, primaryDirection, receiver, shadowEntry);
        !opened)
        return opened;
    record = {
        .origin = origin.getPtr(),
        .direction = direction.getPtr(),
        .throughput = throughput.getPtr(),
        .radiance = radiance.getPtr(),
        .state = state.getPtr(),
        .hit = hit.getPtr(),
        .hitInstance = hitInstance.getPtr(),
        .shadowOrigin = shadowOrigin.getPtr(),
        .shadowDirection = shadowDirection.getPtr(),
        .shadowRadiance = shadowRadiance.getPtr(),
        .pathBin = pathBin.getPtr(),
        .closures = closures.getPtr(),
        .lightPath = lightPath.getPtr(),
        .minRayPdf = minRayPdf.getPtr(),
        .primaryDirection = primaryDirection.getPtr(),
        .receiver = receiver.getPtr(),
        .capacity = capacity,
        .shadowEntry = shadowEntry.getPtr(),
    };
    return {};
}

}
