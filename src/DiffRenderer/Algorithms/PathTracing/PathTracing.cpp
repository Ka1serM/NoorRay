#include "DiffRenderer/Algorithms/PathTracing/PathTracing.h"

#include <algorithm>
#include <cmath>

namespace DiffRenderer::Algorithms {

namespace {

constexpr float albedoRate = 0.02f;
constexpr float translationRate = 0.01f;
constexpr PathSampling targetSampling{.samples = 1024, .seed = 1};

const Description description{
    .id = "path-tracing",
    .name = "Path tracing",
    .representation = "a triangle-mesh sphere above a diffuse ground plane, lit by a rectangular area light",
    .parameters = "object albedo, ground albedo, object translation",
    .discontinuities = "edge sampling of the mesh edges adds the boundary terms of the object silhouette in the image and of the shadow edges on the light",
    .reference = "Mitsuba 3 direct_projective",
    .hasBoundaryTerm = true,
};

}

const Description& PathTracing::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> PathTracing::open(NoorRHI::Device& device, const Settings& settings) {
    Core::Random random(settings.seed);
    scene.emplace(random);
    parameters = scene->perturb(random);
    boundaryTerm = settings.boundaryTerm;
    iteration = 0;
    if (std::expected<void, NoorRHI::Error> opened = tracer.open(device, *scene); !opened)
        return opened;
    target = Core::Image(PathTracer::width, PathTracer::height);
    current = Core::Image(PathTracer::width, PathTracer::height);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    albedoOptimizer.reset();
    albedoOptimizer.setLearningRate(albedoRate * settings.learningRateScale);
    translationOptimizer.reset();
    translationOptimizer.setLearningRate(translationRate * settings.learningRateScale);
    return tracer.render(scene->truth, targetSampling, target.getPixels());
}

std::expected<Step, NoorRHI::Error> PathTracing::step() {
    const std::uint32_t seed = 2 * iteration + 5000;
    ++iteration;
    if (std::expected<void, NoorRHI::Error> rendered = tracer.render(parameters, {.samples = 16, .seed = seed}, current.getPixels()); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    const PathSampling sampling{.samples = 16, .lightSamples = 4, .edgeSamples = 512, .seed = seed + 1, .boundary = boundaryTerm};
    if (std::expected<void, NoorRHI::Error> derived = tracer.backward(imageGradient, sampling, gradients); !derived)
        return std::unexpected(std::move(derived.error()));
    albedoOptimizer.step(parameters.albedos, gradients.albedos);
    translationOptimizer.step(parameters.translation, gradients.translation);
    for (float& albedo : parameters.albedos)
        albedo = std::clamp(albedo, 0.0f, 1.0f);
    double error = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        error += std::abs(double{parameters.translation[axis]} - scene->truth.translation[axis]);
    return Step{.loss = loss, .parameterError = error / 3.0};
}

}
