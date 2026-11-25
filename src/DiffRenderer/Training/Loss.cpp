#include "DiffRenderer/Training/Loss.h"

#include <array>
#include <utility>

#include "ImageLoss.h"
#include "ImageLoss.slang.h"

namespace DiffRenderer::Training {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t groupWidth = 16;
constexpr std::size_t momentCount = 5;
constexpr std::size_t partialCount = 3;

std::uint32_t getGroups(const std::uint32_t size) {
    return (size + groupWidth - 1) / groupWidth;
}

}

std::expected<void, Error> Loss::open(Device& target, const std::uint32_t width, const std::uint32_t height, const LossKind chosen) {
    if (width == 0 || height == 0)
        return fail(ErrorCode::InvalidArgument, "the loss needs an image size above zero");
    device = &target;
    kind = chosen;
    maxWidth = width;
    maxHeight = height;
    accumulating = false;
    oversized = false;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 6>{{{&filterMomentRows, &ImageLoss::filterMomentRows},
                                                                                              {&filterMomentColumns, &ImageLoss::filterMomentColumns},
                                                                                              {&filterPartialRows, &ImageLoss::filterPartialRows},
                                                                                              {&filterPartialColumns, &ImageLoss::filterPartialColumns},
                                                                                              {&squaredError, &ImageLoss::squaredError},
                                                                                              {&sumLosses, &ImageLoss::sumLosses}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    const std::size_t planes = std::size_t{width} * height * 3;
    for (std::expected<void, Error> step :
         {rows.open(target, planes * momentCount), partials.open(target, planes * partialCount),
          groupLosses.open(target, std::size_t{getGroups(width)} * getGroups(height)), value.open(target, 1), readback.open(target, 1)})
        if (!step)
            return step;
    return {};
}

void Loss::record(CommandList& list, const Ptr<float> image, const Ptr<uint> target, const Ptr<float> gradient, const std::uint32_t width,
                  const std::uint32_t height) {
    if (width > maxWidth || height > maxHeight) {
        oversized = true;
        return;
    }
    if (!accumulating)
        list.upload(value, std::span<const float>(std::array{0.0f}));
    accumulating = true;
    const std::uint32_t groupCount = getGroups(width) * getGroups(height);
    const ImageLoss::Arguments arguments{image,
                                         target,
                                         gradient,
                                         rows.getPtr(),
                                         partials.getPtr(),
                                         groupLosses.getPtr(),
                                         value.getPtr(),
                                         width,
                                         height,
                                         groupCount,
                                         1.0f / (static_cast<float>(width) * static_cast<float>(height) * 3.0f)};
    const DispatchSize pixels{width, height};
    if (kind == LossKind::L2) {
        list.dispatch(squaredError, pixels, arguments);
    } else {
        list.dispatch(filterMomentRows, pixels, arguments);
        list.dispatch(filterMomentColumns, pixels, arguments);
        list.dispatch(filterPartialRows, pixels, arguments);
        list.dispatch(filterPartialColumns, pixels, arguments);
    }
    list.dispatch(sumLosses, {groupWidth * groupWidth}, arguments);
}

void Loss::recordReadback(CommandList& list) {
    list.download(value, readback);
    accumulating = false;
}

std::expected<double, Error> Loss::read() {
    if (std::exchange(oversized, false))
        return fail(ErrorCode::InvalidArgument, "an image of the loss is larger than the size that the loss opened with");
    std::expected<std::span<const float>, Error> values = readback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    return double{values->front()};
}

}
