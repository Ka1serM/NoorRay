#include "DiffRenderer/Algorithms/GaussianSplatting/GaussianSplatting.h"

#include <cmath>

namespace DiffRenderer::Algorithms {

namespace {

constexpr std::array<float, 5> learningRates{0.002f, 0.005f, 0.001f, 0.01f, 0.01f};

const Description description{
    .id = "gaussian-splatting",
    .name = "Gaussian splatting",
    .representation = "300 anisotropic 3D Gaussians",
    .parameters = "positions, rotations, scales, opacities, colours",
    .discontinuities = "none: every Gaussian is smooth, so the image is differentiable everywhere",
    .reference = "gsplat",
};

double getMeanError(const std::vector<float>& values, const std::vector<float>& expected) {
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index)
        sum += std::abs(double{values[index]} - expected[index]);
    return sum / static_cast<double>(values.size());
}

}

const Description& GaussianSplatting::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> GaussianSplatting::open(NoorRHI::Device& device, const Settings& settings) {
    Core::Random random(settings.seed);
    truth = makeRandomGaussians(count, random);
    parameters = perturbGaussians(*truth, random);
    gradients.emplace(count);
    if (std::expected<void, NoorRHI::Error> opened = splatter.open(device, count, camera); !opened)
        return opened;
    target = Core::Image(camera.width, camera.height);
    current = Core::Image(camera.width, camera.height);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    for (std::size_t group = 0; group < optimizers.size(); ++group) {
        optimizers[group].reset();
        optimizers[group].setLearningRate(learningRates[group] * settings.learningRateScale);
    }
    return splatter.render(*truth, target.getPixels());
}

std::expected<Step, NoorRHI::Error> GaussianSplatting::step() {
    if (std::expected<void, NoorRHI::Error> rendered = splatter.render(*parameters, current.getPixels()); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    if (std::expected<void, NoorRHI::Error> derived = splatter.backward(imageGradient, *gradients); !derived)
        return std::unexpected(std::move(derived.error()));
    for (std::size_t group = 0; group < optimizers.size(); ++group)
        optimizers[group].step(*parameters->getGroups()[group], *gradients->getGroups()[group]);
    clampGaussians(*parameters);
    return Step{.loss = loss, .parameterError = getMeanError(parameters->means, truth->means)};
}

}
