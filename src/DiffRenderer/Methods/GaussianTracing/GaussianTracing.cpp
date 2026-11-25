#include "DiffRenderer/Methods/GaussianTracing/GaussianTracing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "GaussianRays.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

const Description description{
    .id = "3dgrt",
    .name = "Spectral Gaussian ray tracing",
    .representation = "3D Gaussians, SH degree 3",
    .renderer = "ray tracing of proxy meshes, composited by peak in linear radiance as the NoorRay path tracer",
    .discontinuities = "none in the alpha blending; the order of the peaks and the cutoff at alpha 1/255 have no boundary term",
    .reference = "Moenne-Loccoz et al. 2024, 3D Gaussian Ray Tracing",
    .scene = true,
    .object = true,
    .volume = true,
};

constexpr std::uint32_t gradientStride = 59;
//! The GPU watchdog limits the time of one submission, so each band of rows is its own submission; a small band leaves the GPU waiting for its deepest rays.
constexpr std::uint32_t pixelsPerDispatch = 1u << 17;

float srgbToLinear(const float encoded) {
    return encoded <= 0.04045f ? encoded / 12.92f : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

}

const Description& GaussianTracing::getDescription() const {
    return description;
}

std::expected<void, Error> GaussianTracing::open(Device& target, const Scenes::Dataset& dataset, const Settings& chosen) {
    close();
    device = &target;
    if (std::expected<void, Error> opened = model.open(target, dataset, chosen); !opened)
        return opened;
    background = {srgbToLinear(dataset.background.x), srgbToLinear(dataset.background.y), srgbToLinear(dataset.background.z), 0.0f};
    for (const auto* views : {&dataset.train, &dataset.test})
        for (const Scenes::View& view : *views)
            maxPixels = std::max(maxPixels, view.camera.width * view.camera.height);
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 4>{{{&clearGradients, &GaussianRays::clearGradients},
                                                                                              {&traceForward, &GaussianRays::traceForward},
                                                                                              {&traceBackward, &GaussianRays::traceBackward},
                                                                                              {&gatherGradients, &GaussianRays::gatherGradients}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    for (std::expected<void, Error> step : {image.open(target, std::size_t{maxPixels} * 3), radiance.open(target, std::size_t{maxPixels} * 3),
                                            imageGradient.open(target, std::size_t{maxPixels} * 3), imageReadback.open(target, std::size_t{maxPixels} * 3)})
        if (!step)
            return step;
    return openWorkBuffers(model.getCapacity());
}

std::expected<void, Error> GaussianTracing::openWorkBuffers(const std::uint32_t chosen) {
    workCapacity = chosen;
    if (std::expected<void, Error> opened = rayGradients.open(*device, std::size_t{workCapacity} * gradientStride); !opened)
        return opened;
    return proxies.open(*device, workCapacity);
}

std::expected<void, Error> GaussianTracing::submit(CommandList&& list) const {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

GaussianRays::Parameters GaussianTracing::getPointers(Buffer<float> GaussianModel::GroupBuffers::* member) const {
    const auto get = [&](const GaussianModel::Group group) { return (model.getGroup(group).*member).getPtr(); };
    return {get(GaussianModel::Means),     get(GaussianModel::FeaturesDc), get(GaussianModel::FeaturesRest),
            get(GaussianModel::Opacities), get(GaussianModel::Scales),     get(GaussianModel::Rotations)};
}

std::expected<void, Error> GaussianTracing::rebuildProxies() {
    const GaussianRays::Parameters parameters = getPointers(&GaussianModel::GroupBuffers::parameters);
    Instances::Arguments instances{};
    instances.means = parameters.means;
    instances.quaternions = parameters.rotations;
    instances.logScales = parameters.scales;
    instances.opacityLogits = parameters.opacities;
    instances.count = model.getCount();
    CommandList list(*device);
    proxies.recordInstances(list, instances);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    return proxies.rebuild(model.getCount());
}

std::expected<GaussianRays::Arguments, Error> GaussianTracing::getArguments(const Scenes::Camera& camera, const bool accumulate) const {
    if (camera.width * camera.height > maxPixels)
        return fail(ErrorCode::InvalidArgument, "the camera is larger than the views that the Gaussian ray tracing opened with");
    const float3 position = camera.getPosition();
    GaussianRays::Arguments arguments{};
    arguments.view = {{camera.worldToCamera[0], camera.worldToCamera[1], camera.worldToCamera[2]},
                      camera.intrinsics,
                      {position.x, position.y, position.z, 0.0f},
                      background,
                      camera.width,
                      camera.height,
                      0,
                      camera.height};
    arguments.parameters = getPointers(&GaussianModel::GroupBuffers::parameters);
    arguments.gradients = getPointers(&GaussianModel::GroupBuffers::gradients);
    arguments.rayGradients = rayGradients.getPtr();
    arguments.tlas = proxies.getTlas();
    arguments.image = image.getPtr();
    arguments.radiance = radiance.getPtr();
    arguments.imageGradient = imageGradient.getPtr();
    arguments.gradientSums = model.getGradientSums();
    arguments.gradientCounts = model.getGradientCounts();
    arguments.count = model.getCount();
    arguments.threads = GaussianModel::getThreads(std::uint64_t{model.getCount()} * gradientStride);
    arguments.shDegree = model.getShDegree();
    arguments.accumulate = accumulate ? 1u : 0u;
    return arguments;
}

std::expected<void, Error> GaussianTracing::submitRows(const ComputePipeline& pipeline, const GaussianRays::Arguments& arguments) const {
    GaussianRays::Arguments band = arguments;
    const std::uint32_t rowsPerDispatch = std::max((pixelsPerDispatch / arguments.view.width) & ~7u, 8u);
    for (std::uint32_t row = 0; row < arguments.view.height; row += rowsPerDispatch) {
        band.view.firstRow = row;
        band.view.rows = std::min(rowsPerDispatch, arguments.view.height - row);
        CommandList list(*device);
        list.dispatch(pipeline, {arguments.view.width, band.view.rows}, band);
        if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
            return submitted;
    }
    return {};
}

std::expected<double, Error> GaussianTracing::step(const Target& target, Training::Loss& loss) {
    const GaussianModel::Phase phase = model.beginStep();
    const Scenes::Camera& camera = target.view->camera;
    if (std::expected<void, Error> rebuilt = rebuildProxies(); !rebuilt)
        return std::unexpected(std::move(rebuilt.error()));
    std::expected<GaussianRays::Arguments, Error> arguments = getArguments(camera, phase.densifying);
    if (!arguments)
        return std::unexpected(std::move(arguments.error()));
    GaussianRays::Arguments perGaussian = *arguments;
    perGaussian.threads = GaussianModel::getThreads(model.getCount());
    CommandList clearing(*device);
    clearing.dispatch(clearGradients, {arguments->threads}, *arguments);
    if (std::expected<void, Error> submitted = submit(std::move(clearing)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    if (std::expected<void, Error> traced = submitRows(traceForward, *arguments); !traced)
        return std::unexpected(std::move(traced.error()));
    CommandList scoring(*device);
    loss.record(scoring, image.getPtr(), target.image, imageGradient.getPtr(), camera.width, camera.height);
    if (std::expected<void, Error> submitted = submit(std::move(scoring)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    if (std::expected<void, Error> traced = submitRows(traceBackward, *arguments); !traced)
        return std::unexpected(std::move(traced.error()));
    CommandList updating(*device);
    updating.dispatch(gatherGradients, {perGaussian.threads}, perGaussian);
    model.recordUpdate(updating);
    loss.recordReadback(updating);
    if (std::expected<void, Error> submitted = submit(std::move(updating)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    std::expected<double, Error> value = loss.read();
    if (!value)
        return value;
    if (std::expected<void, Error> finished = model.finishStep(phase); !finished)
        return std::unexpected(std::move(finished.error()));
    if (model.getCapacity() > workCapacity)
        if (std::expected<void, Error> opened = openWorkBuffers(model.getCapacity()); !opened)
            return std::unexpected(std::move(opened.error()));
    return value;
}

std::expected<void, Error> GaussianTracing::render(const Scenes::Camera& camera, Core::Image& result) {
    if (std::expected<void, Error> rebuilt = rebuildProxies(); !rebuilt)
        return rebuilt;
    std::expected<GaussianRays::Arguments, Error> arguments = getArguments(camera, false);
    if (!arguments)
        return std::unexpected(std::move(arguments.error()));
    if (std::expected<void, Error> traced = submitRows(traceForward, *arguments); !traced)
        return traced;
    CommandList list(*device);
    list.download(image, imageReadback);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    std::expected<std::span<const float>, Error> pixels = imageReadback.read();
    if (!pixels)
        return std::unexpected(std::move(pixels.error()));
    const std::span<float> destination = result.getPixels();
    for (std::size_t index = 0; index < destination.size(); ++index)
        destination[index] = std::clamp((*pixels)[index], 0.0f, 1.0f);
    return {};
}

std::expected<std::optional<TrainedGaussians>, Error> GaussianTracing::readGaussians() {
    return model.readGaussians();
}

}
