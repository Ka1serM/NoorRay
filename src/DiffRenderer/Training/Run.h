#pragma once

#include <expected>
#include <span>
#include <vector>

#include "DiffRenderer/Algorithms/Algorithm.h"

namespace DiffRenderer::Training {

struct Sample {
    int iteration = 0;
    double loss = 0.0;
    double psnr = 0.0;
    double parameterError = 0.0;
    double milliseconds = 0.0;
};

//! One optimization of one algorithm, with the history of every step.
class Run {
public:
    [[nodiscard]] std::expected<void, NoorRHI::Error> open(Algorithms::Algorithm& algorithm, NoorRHI::Device& device, const Algorithms::Settings& settings);
    [[nodiscard]] std::expected<void, NoorRHI::Error> step();

    bool isOpen() const { return algorithm != nullptr; }
    Algorithms::Algorithm& getAlgorithm() const { return *algorithm; }
    const Algorithms::Settings& getSettings() const { return settings; }
    std::span<const Sample> getSamples() const { return samples; }
    int getIteration() const { return static_cast<int>(samples.size()); }
    double getOpenMilliseconds() const { return openMilliseconds; }
    double getMedianStepMilliseconds() const;

private:
    Algorithms::Algorithm* algorithm = nullptr;
    Algorithms::Settings settings;
    std::vector<Sample> samples;
    double openMilliseconds = 0.0;
};

}
