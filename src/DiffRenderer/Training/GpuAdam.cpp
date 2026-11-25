#include "DiffRenderer/Training/GpuAdam.h"

#include <algorithm>
#include <cmath>

#include "Optimizing.h"
#include "Optimizing.slang.h"

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

//! A dispatch has at most 65535 groups in x; larger arrays loop with a stride of the thread count.
constexpr std::uint32_t maxThreads = 65535u * 256u;

std::uint32_t getThreads(const std::uint64_t count) {
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(count, 1, maxThreads));
}

}

std::expected<void, Error> GpuAdam::open(Device& device, const AdamSettings& chosen) {
    settings = chosen;
    if (std::expected<void, Error> opened = adam.open(device, Optimizing::adam); !opened)
        return opened;
    return gather.open(device, Optimizing::gather);
}

void GpuAdam::record(CommandList& list, const AdamGroup& group, const std::uint32_t step) const {
    const double firstCorrection = 1.0 - std::pow(double{settings.beta1}, step);
    const double secondCorrection = 1.0 - std::pow(double{settings.beta2}, step);
    recordCorrected(list, group, static_cast<float>(group.learningRate / firstCorrection), static_cast<float>(1.0 / std::sqrt(secondCorrection)));
}

void GpuAdam::recordCorrected(CommandList& list, const AdamGroup& group, const float stepSize, const float secondCorrection) const {
    const std::uint32_t threads = getThreads(group.count);
    list.dispatch(adam, {threads},
                  Optimizing::AdamArguments{group.parameters, group.gradients, group.first, group.second, group.count, threads, stepSize, settings.beta1,
                                            settings.beta2, settings.epsilon, secondCorrection});
}

void GpuAdam::recordGather(CommandList& list, const Ptr<float> source, const Ptr<float> destination, const Ptr<uint> indices, const std::uint32_t count,
                           const std::uint32_t stride) const {
    const std::uint32_t threads = getThreads(std::uint64_t{count} * stride);
    list.dispatch(gather, {threads}, Optimizing::GatherArguments{source, destination, indices, count, threads, stride});
}

}
