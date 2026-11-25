#include "DiffRenderer/Algorithms/PathTracing/PathTracer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "DirectLighting.slang.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr std::size_t pixelCount = std::size_t{PathTracer::width} * PathTracer::height;
constexpr std::array<Atomic<float>, DirectLighting::parameterCount> zeroGradients{};

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

template <class T>
std::expected<void, Error> openWith(Device& device, Buffer<T>& buffer, const std::vector<T>& values) {
    if (std::expected<void, Error> opened = buffer.open(device, values.size()); !opened)
        return opened;
    return buffer.upload(values);
}

std::expected<void, Error> buildBlas(Device& device, AccelerationStructure& blas, const Buffer<float3>& positions, const Buffer<std::uint32_t>& indices) {
    const TriangleGeometry geometry{
        .positions = positions.getPtr(), .indices = indices.getPtr(), .triangleCount = static_cast<std::uint32_t>(indices.getSize() / 3)};
    return blas.build(device, {&geometry, 1}, BuildMode::Static);
}

std::array<float, DirectLighting::parameterCount> flatten(const PathParameters& parameters) {
    std::array<float, DirectLighting::parameterCount> values{};
    std::ranges::copy(parameters.albedos, values.begin());
    std::ranges::copy(parameters.translation, values.begin() + 6);
    return values;
}

}

std::expected<void, Error> PathTracer::open(Device& target, const PathScene& scene) {
    device = &target;
    if (!target.getFeatures().hardwareRayTracing)
        return fail(ErrorCode::Unsupported, "the device has no hardware ray tracing");
    for (std::expected<void, Error> step :
         {renderPipeline.open(target, DirectLighting::renderMain), gradientPipeline.open(target, DirectLighting::gradientMain),
          boundaryPipeline.open(target, DirectLighting::primaryBoundaryMain), openWith(target, groundPositions, scene.ground.positions),
          openWith(target, groundIndices, scene.ground.indices), openWith(target, objectPositions, scene.object.positions),
          openWith(target, objectIndices, scene.object.indices), openWith(target, objectNormals, scene.object.normals),
          openWith(target, objectEdges, scene.edges), parameterBuffer.open(target, DirectLighting::parameterCount), weightBuffer.open(target, 3 * pixelCount),
          imageBuffer.open(target, 3 * pixelCount), gradientBuffer.open(target, DirectLighting::parameterCount), imageReadback.open(target, 3 * pixelCount),
          gradientReadback.open(target, DirectLighting::parameterCount)})
        if (!step)
            return step;
    for (std::expected<void, Error> step :
         {buildBlas(target, groundBlas, groundPositions, groundIndices), buildBlas(target, objectBlas, objectPositions, objectIndices)})
        if (!step)
            return step;
    instances[0] = {.blas = &groundBlas, .customIndex = DirectLighting::groundInstance};
    instances[1] = {.blas = &objectBlas, .customIndex = DirectLighting::objectInstance};
    for (Instance& instance : instances)
        for (std::size_t index = 0; index < 4; ++index)
            instance.transform.values[index][index] = 1.0f;
    if (std::expected<void, Error> built = tlas.build(target, instances); !built)
        return built;

    const float3 forward =
        normalize({scene.cameraTarget.x - scene.cameraOrigin.x, scene.cameraTarget.y - scene.cameraOrigin.y, scene.cameraTarget.z - scene.cameraOrigin.z});
    const float3 right = normalize(cross(forward, scene.cameraUp));
    const float tangent = std::tan(scene.fieldOfView * std::numbers::pi_v<float> / 360.0f);
    arguments = {
        .cameraOrigin = extend(scene.cameraOrigin, tangent),
        .cameraForward = extend(forward),
        .cameraRight = extend(right),
        .cameraUp = extend(cross(right, forward)),
        .lightCorner = extend(scene.lightCorner),
        .lightEdgeU = extend(scene.lightEdgeU),
        .lightEdgeV = extend(scene.lightEdgeV),
        .lightRadiance = extend(scene.lightRadiance),
        .scene = tlas.getHandle(),
        .parameters = parameterBuffer.getPtr(),
        .groundPositions = groundPositions.getPtr().as<float>(),
        .groundIndices = groundIndices.getPtr(),
        .objectPositions = objectPositions.getPtr().as<float>(),
        .objectIndices = objectIndices.getPtr(),
        .objectNormals = objectNormals.getPtr().as<float>(),
        .objectEdges = objectEdges.getPtr(),
        .weights = weightBuffer.getPtr(),
        .image = imageBuffer.getPtr(),
        .gradients = gradientBuffer.getPtr(),
        .width = width,
        .height = height,
        .edgeCount = static_cast<std::uint32_t>(scene.edges.size()),
    };
    return {};
}

std::expected<void, Error> PathTracer::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> PathTracer::render(const PathParameters& parameters, const PathSampling& sampling, const std::span<float> result) {
    const std::array<float, DirectLighting::parameterCount> values = flatten(parameters);
    if (std::expected<void, Error> uploaded = parameterBuffer.upload(values); !uploaded)
        return uploaded;
    for (std::size_t axis = 0; axis < 3; ++axis)
        instances[1].transform.values[axis][3] = parameters.translation[axis];
    if (std::expected<void, Error> updated = tlas.update(instances); !updated)
        return updated;
    DirectLighting::Arguments renderArguments = arguments;
    renderArguments.samples = sampling.samples;
    renderArguments.seed = sampling.seed;
    renderArguments.lightSamples = sampling.lightSamples;
    CommandList list(*device);
    list.dispatch(renderPipeline, {width, height}, renderArguments);
    list.download(imageBuffer, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> image = imageReadback.read();
    if (!image)
        return std::unexpected(std::move(image.error()));
    std::ranges::copy(*image, result.begin());
    return {};
}

std::expected<void, Error> PathTracer::backward(const std::span<const float> imageGradient, const PathSampling& sampling, PathParameters& result) {
    DirectLighting::Arguments gradientArguments = arguments;
    gradientArguments.samples = sampling.samples;
    gradientArguments.seed = sampling.seed;
    gradientArguments.edgeSamples = sampling.edgeSamples;
    gradientArguments.lightSamples = sampling.lightSamples;
    gradientArguments.terms = DirectLighting::interiorTerm | (sampling.boundary ? DirectLighting::shadowBoundaryTerm : 0u);
    CommandList list(*device);
    list.upload(weightBuffer, imageGradient);
    list.upload(gradientBuffer, std::span<const Atomic<float>>(zeroGradients));
    list.dispatch(gradientPipeline, {width, height}, gradientArguments);
    if (sampling.boundary)
        list.dispatch(boundaryPipeline, {gradientArguments.edgeCount * sampling.edgeSamples}, gradientArguments);
    list.download(gradientBuffer, gradientReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const Atomic<float>>, Error> gradients = gradientReadback.read();
    if (!gradients)
        return std::unexpected(std::move(gradients.error()));
    for (std::size_t index = 0; index < 6; ++index)
        result.albedos[index] = (*gradients)[index].value;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.translation[axis] = (*gradients)[6 + axis].value;
    return {};
}

}
