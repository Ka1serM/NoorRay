#pragma once

#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/Rasterization/Rasterizer.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

class Rasterization final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    TexturedCube cube;
    Rasterizer rasterizer;
    std::vector<float> positions;
    std::vector<float> gradients;
    std::vector<NoorRHI::float4> clip;
    std::vector<NoorRHI::float4> clipGradient;
    Core::Adam optimizer;
    Core::Image target;
    Core::Image current;
    std::vector<float> imageGradient;
};

}
