#include "DiffRenderer/Algorithms/SdfTracing/SdfTracer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "SphereTracing.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr std::size_t pixelCount = std::size_t{SdfTracer::width} * SdfTracer::height;
constexpr std::size_t parameterCount = SphereTracing::gridOffset + SdfScene::gridSize;

float3 normalize(const float3 value) {
    const float length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    return {value.x / length, value.y / length, value.z / length};
}

float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float4 extend(const float3 value, const float w = 0.0f) {
    return {value.x, value.y, value.z, w};
}

}

std::expected<void, Error> SdfTracer::open(Device& target, const SdfScene& scene) {
    device = &target;
    for (std::expected<void, Error> step :
         {renderPipeline.open(target, SphereTracing::renderMain), interiorPipeline.open(target, SphereTracing::interiorMain),
          boundaryPipeline.open(target, SphereTracing::boundaryMain), parameterBuffer.open(target, parameterCount), weightBuffer.open(target, 3 * pixelCount),
          imageBuffer.open(target, 3 * pixelCount), gradientBuffer.open(target, parameterCount), imageReadback.open(target, 3 * pixelCount),
          gradientReadback.open(target, parameterCount)})
        if (!step)
            return step;
    const std::array<float, SphereTracing::gridOffset> pose{scene.albedo[0], scene.albedo[1], scene.albedo[2], 0.0f, 0.0f, 0.0f};
    if (std::expected<void, Error> uploaded = parameterBuffer.upload(pose); !uploaded)
        return uploaded;
    zeroGradients.assign(parameterCount, {});
    for (std::size_t view = 0; view < views.size(); ++view) {
        const SdfCamera& camera = scene.views[view];
        const float3 forward = normalize({camera.target.x - camera.origin.x, camera.target.y - camera.origin.y, camera.target.z - camera.origin.z});
        const float3 right = normalize(cross(forward, camera.up));
        const std::size_t tileOffset = 3 * (std::size_t{view / columns} * tileSize * width + std::size_t{view % columns} * tileSize);
        views[view] = {
            .cameraOrigin = extend(camera.origin, std::tan(camera.fieldOfView * std::numbers::pi_v<float> / 360.0f)),
            .cameraForward = extend(forward),
            .cameraRight = extend(right),
            .cameraUp = extend(cross(right, forward)),
            .light = extend(scene.lightDirection, scene.ambient),
            .parameters = parameterBuffer.getPtr(),
            .weights = weightBuffer.getPtr() + tileOffset,
            .image = imageBuffer.getPtr() + tileOffset,
            .gradients = gradientBuffer.getPtr(),
            .width = tileSize,
            .height = tileSize,
            .resolution = SdfScene::resolution,
            .pitch = width,
        };
    }
    return {};
}

SphereTracing::Arguments SdfTracer::getArguments(const std::size_t view, const SdfSampling& sampling) const {
    SphereTracing::Arguments arguments = views[view];
    arguments.samples = sampling.samples;
    arguments.seed = sampling.seed * static_cast<std::uint32_t>(views.size()) + static_cast<std::uint32_t>(view);
    arguments.linesPerPixel = 1;
    return arguments;
}

std::expected<void, Error> SdfTracer::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> SdfTracer::render(const std::span<const float> grid, const SdfSampling& sampling, const std::span<float> result) {
    CommandList list(*device);
    list.upload(parameterBuffer, grid, SphereTracing::gridOffset);
    for (std::size_t view = 0; view < views.size(); ++view)
        list.dispatch(renderPipeline, {tileSize, tileSize}, getArguments(view, sampling));
    list.download(imageBuffer, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> image = imageReadback.read();
    if (!image)
        return std::unexpected(std::move(image.error()));
    std::ranges::copy(*image, result.begin());
    return {};
}

std::expected<void, Error> SdfTracer::backward(const std::span<const float> imageGradient, const SdfSampling& sampling, const std::span<float> gridGradient) {
    CommandList list(*device);
    list.upload(weightBuffer, imageGradient);
    list.upload(gradientBuffer, std::span<const Atomic<float>>(zeroGradients));
    for (std::size_t view = 0; view < views.size(); ++view) {
        const SphereTracing::Arguments arguments = getArguments(view, sampling);
        list.dispatch(interiorPipeline, {tileSize, tileSize}, arguments);
        if (sampling.boundary)
            list.dispatch(boundaryPipeline, {tileSize, 2 * tileSize * arguments.linesPerPixel}, arguments);
    }
    list.download(gradientBuffer, gradientReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> gradients = gradientReadback.read();
    if (!gradients)
        return std::unexpected(std::move(gradients.error()));
    for (std::size_t index = 0; index < gridGradient.size(); ++index)
        gridGradient[index] = (*gradients)[SphereTracing::gridOffset + index].value;
    return {};
}

}
