#include "DiffRenderer/Algorithms/XrayProjection/XrayProjection.h"

#include <algorithm>
#include <cmath>

#include "DiffRenderer/Core/Random.h"

namespace DiffRenderer::Algorithms {

namespace {

constexpr std::uint32_t volumeSize = 96;
constexpr float learningRate = 0.01f;
constexpr float attenuation = 2.5f;
constexpr Pose truePose{0.05f, -0.1f, 0.6f, 0.0f, 0.0f, -1.5f};
constexpr Pose poseOffset{0.1f, 0.08f, 0.15f, 0.05f, 0.04f, 0.08f};

const Description description{
    .id = "xray-projection",
    .name = "X-ray projection",
    .representation = "96^3 head phantom, digitally reconstructed radiograph",
    .parameters = "6-DoF pose of the source and detector (2D/3D registration)",
    .discontinuities = "none: the line integral of the trilinear density is smooth in the pose",
    .reference = "DiffDRR",
};

//! Shows the attenuation 1 - exp(-mu L), so dense bone is bright as on a radiograph.
void toAttenuation(const std::span<const float> lineIntegrals, Core::Image& image) {
    const std::span<float> pixels = image.getPixels();
    for (std::size_t pixel = 0; pixel < lineIntegrals.size(); ++pixel)
        std::fill_n(pixels.begin() + static_cast<std::ptrdiff_t>(pixel * 3), 3, 1.0f - std::exp(-attenuation * lineIntegrals[pixel]));
}

}

const Description& XrayProjection::getDescription() const {
    return description;
}

std::expected<void, NoorRHI::Error> XrayProjection::open(NoorRHI::Device& device, const Settings& settings) {
    const Phantom phantom(volumeSize);
    if (std::expected<void, NoorRHI::Error> opened = projector.open(device, phantom); !opened)
        return opened;
    Core::Random random(settings.seed);
    truth = truePose;
    for (std::size_t index = 0; index < pose.size(); ++index)
        pose[index] = truth[index] + (random.uniform(0.0f, 1.0f) < 0.5f ? -poseOffset[index] : poseOffset[index]);
    target = Core::Image(Projector::imageSize, Projector::imageSize);
    current = Core::Image(Projector::imageSize, Projector::imageSize);
    lineIntegrals.assign(std::size_t{Projector::imageSize} * Projector::imageSize, 0.0f);
    lineIntegralGradient.assign(lineIntegrals.size(), 0.0f);
    imageGradient.assign(target.getPixels().size(), 0.0f);
    optimizer.reset();
    optimizer.setLearningRate(learningRate * settings.learningRateScale);
    if (std::expected<void, NoorRHI::Error> rendered = projector.render(truth, lineIntegrals); !rendered)
        return rendered;
    toAttenuation(lineIntegrals, target);
    return {};
}

std::expected<Step, NoorRHI::Error> XrayProjection::step() {
    if (std::expected<void, NoorRHI::Error> rendered = projector.render(pose, lineIntegrals); !rendered)
        return std::unexpected(std::move(rendered.error()));
    toAttenuation(lineIntegrals, current);
    const double loss = Core::getL2Loss(current, target, imageGradient);
    for (std::size_t pixel = 0; pixel < lineIntegrals.size(); ++pixel) {
        const float attenuationGradient = imageGradient[pixel * 3] + imageGradient[pixel * 3 + 1] + imageGradient[pixel * 3 + 2];
        lineIntegralGradient[pixel] = attenuationGradient * attenuation * (1.0f - current.getPixels()[pixel * 3]);
    }
    if (std::expected<void, NoorRHI::Error> derived = projector.backward(lineIntegralGradient, gradient); !derived)
        return std::unexpected(std::move(derived.error()));
    optimizer.step(pose, gradient);
    double error = 0.0;
    for (std::size_t index = 0; index < pose.size(); ++index)
        error += std::abs(double{pose[index]} - truth[index]);
    return Step{.loss = loss, .parameterError = error / static_cast<double>(pose.size())};
}

}
