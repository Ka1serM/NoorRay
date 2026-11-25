#include "DiffRenderer/Algorithms/VolumeRendering/RayMarcher.h"

#include <array>
#include <cmath>
#include <numbers>
#include <utility>

#include "RayMarching.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr std::size_t viewPixels = std::size_t{RayMarcher::viewSize} * RayMarcher::viewSize;
constexpr std::size_t pixelCount = RayMarching::viewCount * viewPixels;
constexpr float cameraDistance = 1.9f;
constexpr std::array<float2, RayMarching::viewCount> viewAngles{float2{30.0f, 20.0f}, float2{120.0f, -10.0f}, float2{210.0f, 25.0f}, float2{300.0f, 5.0f}};

float4 getEye(const float2 angles) {
    const double azimuth = angles.x * std::numbers::pi / 180.0;
    const double elevation = angles.y * std::numbers::pi / 180.0;
    return {static_cast<float>(cameraDistance * std::cos(elevation) * std::sin(azimuth)), static_cast<float>(cameraDistance * std::sin(elevation)),
            static_cast<float>(cameraDistance * std::cos(elevation) * std::cos(azimuth)), 0.0f};
}

}

std::expected<void, Error> RayMarcher::open(Device& target, const Phantom& phantom) {
    device = &target;
    const std::size_t voxels = phantom.getDensity().size();
    for (std::expected<void, Error> step :
         {renderPipeline.open(target, RayMarching::render), backwardPipeline.open(target, RayMarching::backward), density.open(target, voxels),
          tumor.open(target, voxels), transferFunction.open(target, transferFunctionSize), transferFunctionGradients.open(target, transferFunctionSize * 4),
          image.open(target, pixelCount), imageGradients.open(target, pixelCount), imageReadback.open(target, pixelCount),
          gradientReadback.open(target, transferFunctionSize * 4)})
        if (!step)
            return step;
    stagedGradients.assign(pixelCount, {});
    zeros.assign(transferFunctionSize * 4, {});
    arguments = {.density = density.getPtr(),
                 .tumor = tumor.getPtr(),
                 .transferFunction = transferFunction.getPtr(),
                 .transferFunctionGradients = transferFunctionGradients.getPtr(),
                 .image = image.getPtr(),
                 .imageGradients = imageGradients.getPtr(),
                 .eyes = {getEye(viewAngles[0]), getEye(viewAngles[1]), getEye(viewAngles[2]), getEye(viewAngles[3])},
                 .background = {0.08f, 0.09f, 0.12f, 0.0f},
                 .tanHalfFov = static_cast<float>(std::tan(20.0 * std::numbers::pi / 180.0)),
                 .stepSize = 1.0f / 128.0f,
                 .extinctionScale = 60.0f,
                 .transferFunctionSize = transferFunctionSize,
                 .volumeSize = phantom.getSize(),
                 .width = viewSize,
                 .height = viewSize};
    CommandList list(target);
    list.upload(density, std::span(phantom.getDensity()));
    list.upload(tumor, std::span(phantom.getTumor()));
    return submit(std::move(list));
}

std::size_t RayMarcher::getMosaicIndex(const std::size_t pixel) {
    const std::size_t view = pixel / viewPixels;
    const std::size_t y = pixel % viewPixels / viewSize + view / 2 * viewSize;
    const std::size_t x = pixel % viewSize + view % 2 * viewSize;
    return (y * mosaicSize + x) * 3;
}

std::expected<void, Error> RayMarcher::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> RayMarcher::render(const std::span<const float> entries, const std::span<float> mosaic) {
    CommandList list(*device);
    list.upload(transferFunction, std::span(reinterpret_cast<const float4*>(entries.data()), transferFunctionSize));
    list.dispatch(renderPipeline, {viewSize, viewSize, RayMarching::viewCount}, arguments);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float4>, Error> pixels = imageReadback.read();
    if (!pixels)
        return std::unexpected(std::move(pixels.error()));
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        const float4 color = (*pixels)[pixel];
        const std::size_t index = getMosaicIndex(pixel);
        mosaic[index] = color.x;
        mosaic[index + 1] = color.y;
        mosaic[index + 2] = color.z;
    }
    return {};
}

std::expected<void, Error> RayMarcher::backward(const std::span<const float> mosaicGradient, const std::span<float> transferFunctionGradient) {
    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel) {
        const std::size_t index = getMosaicIndex(pixel);
        stagedGradients[pixel] = {mosaicGradient[index], mosaicGradient[index + 1], mosaicGradient[index + 2], 0.0f};
    }
    CommandList list(*device);
    list.upload(imageGradients, std::span<const float4>(stagedGradients));
    list.upload(transferFunctionGradients, std::span<const Atomic<float>>(zeros));
    list.dispatch(backwardPipeline, {viewSize, viewSize, RayMarching::viewCount}, arguments);
    list.download(transferFunctionGradients, gradientReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> values = gradientReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    for (std::size_t index = 0; index < transferFunctionGradient.size(); ++index)
        transferFunctionGradient[index] = (*values)[index].value;
    return {};
}

}
