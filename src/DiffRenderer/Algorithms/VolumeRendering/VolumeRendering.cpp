#include "DiffRenderer/Algorithms/VolumeRendering/VolumeRendering.h"

#include <algorithm>
#include <cmath>
#include <span>

#include "DiffRenderer/Core/Random.h"

namespace DiffRenderer::Algorithms {

namespace {

constexpr std::uint32_t volumeSize = 96;
constexpr float learningRate = 0.01f;

const Description description{
    .id = "volume-rendering",
    .name = "Direct volume rendering",
    .representation = "96^3 head phantom, emission-absorption ray marching from 4 views",
    .parameters = "32-entry RGBA transfer function",
    .discontinuities = "none: the ray integral over the trilinear density is smooth in the transfer function",
    .reference = "PyTorch autograd of the same discretization",
};

struct Bump {
    double center;
    double width;
    double opacity;
    std::array<double, 3> color;
};

void fillTrueTransferFunction(const std::span<float> values) {
    constexpr std::array<Bump, 7> bumps{Bump{0.12, 0.025, 0.04, {0.3, 0.5, 0.95}}, Bump{0.22, 0.03, 0.12, {0.95, 0.75, 0.6}},
                                        Bump{0.45, 0.04, 0.05, {0.85, 0.8, 0.75}}, Bump{0.53, 0.02, 0.2, {0.95, 0.85, 0.3}},
                                        Bump{0.63, 0.02, 0.5, {0.8, 0.2, 0.15}},   Bump{0.74, 0.02, 0.8, {1.0, 0.25, 0.2}},
                                        Bump{0.95, 0.05, 0.45, {1.0, 1.0, 0.95}}};
    constexpr double baseWeight = 0.05;
    for (std::uint32_t entry = 0; entry < RayMarcher::transferFunctionSize; ++entry) {
        const double density = static_cast<double>(entry) / (RayMarcher::transferFunctionSize - 1);
        std::array<double, 3> color{0.5 * baseWeight, 0.5 * baseWeight, 0.5 * baseWeight};
        double opacity = 0.0;
        double weightSum = baseWeight;
        for (const Bump& bump : bumps) {
            const double weight = std::exp(-0.5 * std::pow((density - bump.center) / bump.width, 2.0));
            opacity += bump.opacity * weight;
            for (std::size_t channel = 0; channel < 3; ++channel)
                color[channel] += weight * bump.color[channel];
            weightSum += weight;
        }
        for (std::size_t channel = 0; channel < 3; ++channel)
            values[entry * 4 + channel] = static_cast<float>(color[channel] / weightSum);
        values[entry * 4 + 3] = density < 0.06 ? 0.0f : static_cast<float>(std::min(opacity, 1.0));
    }
}

}

const Description& VolumeRendering::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> VolumeRendering::open(NoorRHI::Device& device, const Settings& settings) {
    const Phantom phantom(volumeSize);
    if (std::expected<void, NoorRHI::Error> opened = rayMarcher.open(device, phantom); !opened)
        return opened;
    fillTrueTransferFunction(truth);
    Core::Random random(settings.seed);
    for (std::size_t index = 0; index < valueCount; ++index) {
        const bool opacity = index % 4 == 3;
        const float noise = random.uniform(-1.0f, 1.0f);
        if (opacity && truth[index] == 0.0f) {
            parameters[index] = 0.0f;
            continue;
        }
        parameters[index] = std::clamp(truth[index] + (opacity ? 0.2f : 0.3f) * noise, 0.0f, 1.0f);
    }
    target = Core::Image(RayMarcher::mosaicSize, RayMarcher::mosaicSize);
    current = Core::Image(RayMarcher::mosaicSize, RayMarcher::mosaicSize);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    optimizer.reset();
    optimizer.setLearningRate(learningRate * settings.learningRateScale);
    return rayMarcher.render(truth, target.getPixels());
}

std::expected<Step, NoorRHI::Error> VolumeRendering::step() {
    if (std::expected<void, NoorRHI::Error> rendered = rayMarcher.render(parameters, current.getPixels()); !rendered)
        return std::unexpected(std::move(rendered.error()));
    const double loss = Core::getL2Loss(current, target, imageGradient);
    if (std::expected<void, NoorRHI::Error> derived = rayMarcher.backward(imageGradient, gradients); !derived)
        return std::unexpected(std::move(derived.error()));
    optimizer.step(parameters, gradients);
    double error = 0.0;
    for (std::size_t index = 0; index < valueCount; ++index) {
        parameters[index] = std::clamp(parameters[index], 0.0f, 1.0f);
        error += std::abs(double{parameters[index]} - truth[index]);
    }
    return Step{.loss = loss, .parameterError = error / valueCount};
}

}
