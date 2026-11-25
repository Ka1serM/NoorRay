#include "DiffRenderer/Algorithms/GaussianTracing/GaussianTracing.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace DiffRenderer::Algorithms {

namespace {

constexpr std::uint32_t size = 192;
constexpr std::size_t halfValues = std::size_t{size} * size * 3;
constexpr std::array<float, 5> learningRates{0.002f, 0.005f, 0.001f, 0.01f, 0.01f};
constexpr std::array<GaussianShading, 2> shadings{GaussianShading::Emissive, GaussianShading::Scattering};
constexpr GaussianSampling targetSampling{.samples = 2048, .seed = 1};

const Description description{
    .id = "gaussian-tracing",
    .name = "Spectral Gaussian path tracing",
    .representation =
        "300 anisotropic 3D Gaussians with a mesh floor and box, emissive in the upper half and scattering in the lower half, in four wavelengths",
    .parameters = "positions, rotations, scales, opacities, colours",
    .discontinuities = "no boundary terms: changes of the order along a ray, the cutoff at the alpha 1/255, a Gaussian that reaches a path vertex, and mesh "
                       "edges seen from a moving path vertex",
    .reference = "3D Gaussian Ray Tracing, path replay backpropagation",
};

double getMeanError(const std::vector<float>& values, const std::vector<float>& expected) {
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index)
        sum += std::abs(double{values[index]} - expected[index]);
    return sum / static_cast<double>(values.size());
}

}

const Description& GaussianTracing::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> GaussianTracing::render(const Gaussians& gaussians, const GaussianSampling& sampling, Core::Image& image) {
    if (std::expected<void, NoorRHI::Error> uploaded = tracer.setParameters(gaussians); !uploaded)
        return uploaded;
    for (std::size_t half = 0; half < shadings.size(); ++half)
        if (std::expected<void, NoorRHI::Error> rendered = tracer.render(shadings[half], sampling, image.getPixels().subspan(half * halfValues, halfValues));
            !rendered)
            return rendered;
    return {};
}

std::expected<void, NoorRHI::Error> GaussianTracing::open(NoorRHI::Device& device, const Settings& settings) {
    Core::Random random(settings.seed);
    truth = makeRandomGaussians(count, random);
    parameters = perturbGaussians(*truth, random);
    gradients.emplace(count);
    shadingGradients.emplace(count);
    iteration = 0;
    if (std::expected<void, NoorRHI::Error> opened = tracer.open(device, getFloorAndBox(), count, size, size); !opened)
        return opened;
    target = Core::Image(size, 2 * size);
    current = Core::Image(size, 2 * size);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    for (std::size_t group = 0; group < optimizers.size(); ++group) {
        optimizers[group].reset();
        optimizers[group].setLearningRate(learningRates[group] * settings.learningRateScale);
    }
    return render(*truth, targetSampling, target);
}

std::expected<Step, NoorRHI::Error> GaussianTracing::step() {
    const std::uint32_t seed = 2 * iteration + 5000;
    ++iteration;
    if (std::expected<void, NoorRHI::Error> rendered = render(*parameters, {.samples = 16, .seed = seed}, current); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    for (std::vector<float>* group : gradients->getGroups())
        std::ranges::fill(*group, 0.0f);
    for (std::size_t half = 0; half < shadings.size(); ++half) {
        const std::span<const float> weights = std::span<const float>(imageGradient).subspan(half * halfValues, halfValues);
        if (std::expected<void, NoorRHI::Error> derived = tracer.backward(shadings[half], {.samples = 16, .seed = seed + 1}, weights, *shadingGradients);
            !derived)
            return std::unexpected(std::move(derived.error()));
        for (std::size_t group = 0; group < optimizers.size(); ++group)
            std::ranges::transform(*gradients->getGroups()[group], *shadingGradients->getGroups()[group], gradients->getGroups()[group]->begin(), std::plus{});
    }
    for (std::size_t group = 0; group < optimizers.size(); ++group)
        optimizers[group].step(*parameters->getGroups()[group], *gradients->getGroups()[group]);
    clampGaussians(*parameters);
    for (float& color : parameters->colors)
        color = std::clamp(color, 0.0f, 1.0f);
    return Step{.loss = loss, .parameterError = getMeanError(parameters->means, truth->means)};
}

}
