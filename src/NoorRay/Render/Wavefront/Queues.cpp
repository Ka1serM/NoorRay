#include "NoorRay/Render/Wavefront/Queues.h"

namespace NoorRay::Render {

std::expected<void, Core::Error> Queues::open(NoorRHI::Device& device, const std::uint32_t pathCapacity, const std::uint32_t sampleCapacity,
                                              const std::uint32_t binCount) {
    if (pathCapacity == 0 || sampleCapacity == 0 || binCount == 0)
        return Core::fail(Core::ErrorCode::InvalidArgument, "the wavefront queues need paths, samples and bins");
    const auto openAll = [&](const std::uint32_t count, auto&... buffers) -> std::expected<void, Core::Error> {
        std::expected<void, NoorRHI::Error> result;
        ((result = result ? buffers.open(device, count) : result), ...);
        return Core::check(std::move(result));
    };
    if (auto opened = openAll(pathCapacity, active0, active1, shadow0, shadow1, shade, free); !opened)
        return opened;
    if (auto opened = openAll(binCount, binCounts, binCursors, binDispatch); !opened)
        return opened;
    if (auto opened = openAll(sampleCapacity, samples); !opened)
        return opened;
    if (auto opened = openAll(Shaders::counterCount, counters); !opened)
        return opened;
    if (auto opened = openAll(Shaders::gridSlotCount, gridDispatch); !opened)
        return opened;
    record = {
        .counters = counters.getPtr(),
        .binCounts = binCounts.getPtr(),
        .binCursors = binCursors.getPtr(),
        .active0 = active0.getPtr(),
        .active1 = active1.getPtr(),
        .shadow0 = shadow0.getPtr(),
        .shadow1 = shadow1.getPtr(),
        .shade = shade.getPtr(),
        .free = free.getPtr(),
        .samples = samples.getPtr(),
        .binDispatch = binDispatch.getPtr().as<std::uint32_t>(),
        .binCount = binCount,
        .sampleCapacity = sampleCapacity,
        .gridDispatch = gridDispatch.getPtr().as<std::uint32_t>(),
    };
    return {};
}

}
