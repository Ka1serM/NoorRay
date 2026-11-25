#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/GaussianTracing/GaussianTracer.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

//! The upper half of each image has the emissive Gaussians and the lower half the scattering ones.
class GaussianTracing final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    static constexpr std::size_t count = 300;

    [[nodiscard]] std::expected<void, NoorRHI::Error> render(const Gaussians& gaussians, const GaussianSampling& sampling, Core::Image& image);

    GaussianTracer tracer;
    std::optional<Gaussians> truth;
    std::optional<Gaussians> parameters;
    std::optional<Gaussians> gradients;
    std::optional<Gaussians> shadingGradients;
    std::array<Core::Adam, 5> optimizers;
    Core::Image target;
    Core::Image current;
    std::vector<float> imageGradient;
    std::uint32_t iteration = 0;
};

}
