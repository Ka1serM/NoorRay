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
    if (auto opened = openAll(origin, direction, throughput, radiance, state, hit, hitInstance, shadowOrigin, shadowDirection, shadowRadiance, pathBin);
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
        .capacity = capacity,
        .unused = 0,
    };
    return {};
}

}
