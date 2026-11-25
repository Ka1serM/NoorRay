#include "DiffRenderer/Algorithms/GaussianTracing/GaussianTracer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "GaussianPaths.slang.h"
#include "NoorRay/Spectral/SpectralTables.h"

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

//! A dispatch with more samples would hold the GPU for seconds.
constexpr std::uint32_t samplesPerDispatch = 32;
constexpr std::array<std::size_t, 5> groupOffsets{0, GaussianPaths::quaternionOffset, GaussianPaths::scaleOffset, GaussianPaths::opacityOffset,
                                                  GaussianPaths::colorOffset};

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

std::vector<float4> getSpectralEntries() {
    std::vector<float4> values;
    for (const NoorRay::Shaders::SpectralEntry& entry : NoorRay::Spectral::SpectralTables::buildEntries()) {
        values.push_back(entry.basisAndLambda);
        values.push_back(entry.filmAndPdf);
    }
    return values;
}

}

std::expected<void, Error> GaussianTracer::openMesh(const GaussianScene& scene) {
    for (std::expected<void, Error> step :
         {openWith(*device, positions, scene.positions), openWith(*device, indices, scene.indices), openWith(*device, albedos, scene.albedos)})
        if (!step)
            return step;
    const TriangleGeometry geometry{
        .positions = positions.getPtr(), .indices = indices.getPtr(), .triangleCount = static_cast<std::uint32_t>(scene.indices.size() / 3)};
    if (std::expected<void, Error> built = meshBlas.build(*device, {&geometry, 1}, BuildMode::Static); !built)
        return built;
    Instance instance{.blas = &meshBlas};
    for (std::size_t index = 0; index < 4; ++index)
        instance.transform.values[index][index] = 1.0f;
    return meshTlas.build(*device, {&instance, 1});
}

void GaussianTracer::setScene(const GaussianScene& scene) {
    const float3 forward =
        normalize({scene.cameraTarget.x - scene.cameraOrigin.x, scene.cameraTarget.y - scene.cameraOrigin.y, scene.cameraTarget.z - scene.cameraOrigin.z});
    const float3 right = normalize(cross(forward, scene.cameraUp));
    arguments.cameraOrigin = extend(scene.cameraOrigin, std::tan(scene.fieldOfView * std::numbers::pi_v<float> / 360.0f));
    arguments.cameraForward = extend(forward);
    arguments.cameraRight = extend(right);
    arguments.cameraUp = extend(cross(right, forward));
    arguments.light = extend(scene.lightPosition, scene.lightIntensity);
    arguments.environment = extend(scene.environment, scene.lightTemperature);
}

std::expected<void, Error> GaussianTracer::open(Device& target, const GaussianScene& scene, const std::size_t count, const std::uint32_t width,
                                                const std::uint32_t height) {
    device = &target;
    if (!target.getFeatures().hardwareRayTracing)
        return fail(ErrorCode::Unsupported, "the device has no hardware ray tracing");
    const std::size_t values = std::size_t{width} * height * 3;
    const std::size_t parameterCount = count * GaussianPaths::parameterStride;
    for (std::expected<void, Error> step :
         {renderPipelines[0].open(target, GaussianPaths::renderEmission), renderPipelines[1].open(target, GaussianPaths::renderScattering),
          gradientPipelines[0].open(target, GaussianPaths::differentiateEmission), gradientPipelines[1].open(target, GaussianPaths::differentiateScattering),
          proxies.open(target, static_cast<std::uint32_t>(count)), openWith(target, spectralEntries, getSpectralEntries()),
          parameters.open(target, parameterCount), gradients.open(target, parameterCount), gradientReadback.open(target, parameterCount),
          weights.open(target, values), image.open(target, values), imageReadback.open(target, values), openMesh(scene)})
        if (!step)
            return step;
    arguments = {.mesh = meshTlas.getHandle(),
                 .parameters = parameters.getPtr(),
                 .positions = positions.getPtr().as<float>(),
                 .indices = indices.getPtr(),
                 .albedos = albedos.getPtr().as<float>(),
                 .spectralEntries = spectralEntries.getPtr(),
                 .weights = weights.getPtr(),
                 .image = image.getPtr(),
                 .gradients = gradients.getPtr(),
                 .count = static_cast<uint>(count),
                 .width = width,
                 .height = height};
    setScene(scene);
    return {};
}

std::expected<void, Error> GaussianTracer::setParameters(const Gaussians& gaussians) {
    if (gaussians.getCount() != arguments.count)
        return fail(ErrorCode::InvalidArgument, "the tracer is open for another count of Gaussians");
    const auto groups = gaussians.getGroups();
    for (std::size_t group = 0; group < groups.size(); ++group)
        if (std::expected<void, Error> uploaded = parameters.upload(*groups[group], groupOffsets[group] * arguments.count); !uploaded)
            return uploaded;
    rotations.resize(gaussians.quaternions.size());
    for (std::size_t first = 0; first < rotations.size(); first += 4) {
        const float* const q = &gaussians.quaternions[first];
        const float length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (std::size_t component = 0; component < 4; ++component)
            rotations[first + component] = q[(component + 1) % 4] / length;
    }
    if (std::expected<void, Error> updated =
            proxies.update(*device, gaussians.means, rotations, gaussians.scales, gaussians.opacities, static_cast<std::uint32_t>(gaussians.getCount()));
        !updated)
        return updated;
    arguments.proxies = proxies.getTlas();
    return {};
}

std::expected<void, Error> GaussianTracer::submit(CommandList&& list) {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::expected<void, Error> GaussianTracer::dispatch(const ComputePipeline& pipeline, const GaussianSampling& sampling, const std::uint32_t firstSample) {
    GaussianPaths::Arguments pass = arguments;
    pass.samples = sampling.samples;
    pass.seed = sampling.seed;
    pass.spectral = sampling.spectral ? 1 : 0;
    pass.filterWidth = sampling.filterWidth;
    pass.firstSample = firstSample;
    pass.sampleEnd = std::min(firstSample + samplesPerDispatch, sampling.samples);
    CommandList list(*device);
    list.dispatch(pipeline, {arguments.width, arguments.height}, pass);
    return submit(std::move(list));
}

std::expected<void, Error> GaussianTracer::render(const GaussianShading shading, const GaussianSampling& sampling, const std::span<float> result) {
    for (std::uint32_t first = 0; first < sampling.samples; first += samplesPerDispatch)
        if (std::expected<void, Error> traced = dispatch(renderPipelines[static_cast<std::size_t>(shading)], sampling, first); !traced)
            return traced;
    CommandList list(*device);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> values = imageReadback.read();
    if (!values)
        return std::unexpected(std::move(values.error()));
    std::ranges::copy(*values, result.begin());
    return {};
}

//! A float sum of the samples of all dispatches drops the small terms and is 2 % too low after 8 million of them, so each dispatch has its own sum.
std::expected<void, Error> GaussianTracer::backward(const GaussianShading shading, const GaussianSampling& sampling, const std::span<const float> imageGradient,
                                                    Gaussians& result) {
    if (std::expected<void, Error> uploaded = weights.upload(imageGradient); !uploaded)
        return uploaded;
    std::vector<double> sums(std::size_t{arguments.count} * GaussianPaths::parameterStride);
    for (std::uint32_t first = 0; first < sampling.samples; first += samplesPerDispatch) {
        for (std::expected<void, Error> step : {gradients.clear(), dispatch(gradientPipelines[static_cast<std::size_t>(shading)], sampling, first)})
            if (!step)
                return step;
        CommandList list(*device);
        list.download(gradients, gradientReadback);
        if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
            return submitted;
        std::expected<std::span<const Atomic<float>>, Error> values = gradientReadback.read();
        if (!values)
            return std::unexpected(std::move(values.error()));
        for (std::size_t index = 0; index < sums.size(); ++index)
            sums[index] += static_cast<double>((*values)[index].value);
    }
    const auto groups = result.getGroups();
    for (std::size_t group = 0; group < groups.size(); ++group)
        std::ranges::transform(std::span(sums).subspan(groupOffsets[group] * arguments.count, groups[group]->size()), groups[group]->begin(),
                               [](const double sum) { return static_cast<float>(sum); });
    return {};
}

}
