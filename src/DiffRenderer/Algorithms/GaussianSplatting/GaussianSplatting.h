#pragma once

#include <array>
#include <optional>
#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/GaussianSplatting/Splatter.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

class GaussianSplatting final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    static constexpr std::size_t count = 300;

    SplatCamera camera;
    Splatter splatter;
    std::optional<Gaussians> truth;
    std::optional<Gaussians> parameters;
    std::optional<Gaussians> gradients;
    std::array<Core::Adam, 5> optimizers;
    Core::Image target;
    Core::Image current;
    std::vector<float> imageGradient;
};

}
