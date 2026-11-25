#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"
#include "DiffRenderer/Algorithms/SdfTracing/SdfTracer.h"
#include "DiffRenderer/Core/Adam.h"

namespace DiffRenderer::Algorithms {

class SdfTracing final : public Algorithm {
public:
    const Description& getDescription() const override;
    std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) override;
    std::expected<Step, NoorRHI::Error> step() override;
    const Core::Image& getTarget() const override { return target; }
    const Core::Image& getCurrent() const override { return current; }

private:
    double getBandError() const;

    SdfTracer tracer;
    std::optional<SdfScene> scene;
    std::vector<float> grid;
    std::vector<float> gradient;
    std::vector<std::uint32_t> band;
    Core::Adam optimizer;
    Core::Image target;
    Core::Image current;
    std::vector<float> imageGradient;
    std::uint32_t iteration = 0;
    bool boundaryTerm = true;
};

}
