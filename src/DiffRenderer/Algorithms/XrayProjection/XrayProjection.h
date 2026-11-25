#pragma once

#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/XrayProjection/Projector.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

class XrayProjection final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    Projector projector;
    Pose truth{};
    Pose pose{};
    Pose gradient{};
    Core::Adam optimizer;
    Core::Image target;
    Core::Image current;
    std::vector<float> lineIntegrals;
    std::vector<float> lineIntegralGradient;
    std::vector<float> imageGradient;
};

}
