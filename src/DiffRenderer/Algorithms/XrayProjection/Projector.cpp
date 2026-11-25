#include "DiffRenderer/Algorithms/XrayProjection/Projector.h"

#include <algorithm>
#include <utility>

#include "Radiograph.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr std::size_t pixelCount = std::size_t{Projector::imageSize} * Projector::imageSize;
constexpr std::array<Atomic<float>, 6> zeros{};

}

std::expected<void, Error> Projector::open(Device& target, const Phantom& phantom) {
    device = &target;
    for (std::expected<void, Error> step :
         {renderPipeline.open(target, Radiograph::render), backwardPipeline.open(target, Radiograph::backward),
          density.open(target, phantom.getDensity().size()), image.open(target, pixelCount), imageGradients.open(target, pixelCount),
          poseGradients.open(target, zeros.size()), imageReadback.open(target, pixelCount), poseReadback.open(target, zeros.size())})
        if (!step)
            return step;
    arguments = {.density = density.getPtr(),
                 .image = image.getPtr(),
                 .imageGradients = imageGradients.getPtr(),
                 .poseGradients = poseGradients.getPtr(),
                 .angles = {},
                 .translation = {},
                 .sourceToDetector = 2.5f,
                 .pixelSpacing = 0.01f,
                 .alphaMin = 0.2f,
                 .alphaMax = 1.0f,
                 .pointCount = 320,
                 .volumeSize = phantom.getSize(),
                 .width = imageSize,
                 .height = imageSize};
    CommandList list(target);
    list.upload(density, std::span(phantom.getDensity()));
    return submit(std::move(list));
}

std::expected<void, Error> Projector::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> Projector::render(const Pose& pose, const std::span<float> lineIntegrals) {
    arguments.angles = {pose[0], pose[1], pose[2], 0.0f};
    arguments.translation = {pose[3], pose[4], pose[5], 0.0f};
    CommandList list(*device);
    list.dispatch(renderPipeline, {imageSize, imageSize}, arguments);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> values = imageReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::ranges::copy(*values, lineIntegrals.begin());
    return {};
}

std::expected<void, Error> Projector::backward(const std::span<const float> lineIntegralGradient, Pose& poseGradient) {
    CommandList list(*device);
    list.upload(imageGradients, lineIntegralGradient);
    list.upload(poseGradients, std::span<const Atomic<float>>(zeros));
    list.dispatch(backwardPipeline, {imageSize, imageSize}, arguments);
    list.download(poseGradients, poseReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> values = poseReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    for (std::size_t index = 0; index < poseGradient.size(); ++index)
        poseGradient[index] = (*values)[index].value;
    return {};
}

}
