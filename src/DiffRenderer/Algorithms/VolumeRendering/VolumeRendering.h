#pragma once

#include <array>
#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/VolumeRendering/RayMarcher.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

class VolumeRendering final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    static constexpr std::size_t valueCount = RayMarcher::transferFunctionSize * 4;

    RayMarcher rayMarcher;
    std::array<float, valueCount> truth{};
    std::array<float, valueCount> parameters{};
    std::array<float, valueCount> gradients{};
    Core::Adam optimizer;
    Core::Image target;
    Core::Image current;
    std::vector<float> imageGradient;
};

}
