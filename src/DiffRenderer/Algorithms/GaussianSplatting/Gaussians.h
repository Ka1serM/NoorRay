#pragma once

#include <array>
#include <vector>

#include "DiffRenderer/Core/Random.h"

namespace DiffRenderer::Algorithms {

//! The parameters of the Gaussians in five groups, one array for each group.
struct Gaussians {
    static constexpr std::array<std::size_t, 5> groupSizes{3, 4, 3, 1, 3};

    explicit Gaussians(std::size_t count) : means(count * 3), quaternions(count * 4), scales(count * 3), opacities(count), colors(count * 3) {}

    std::size_t getCount() const { return opacities.size(); }
    std::array<std::vector<float>*, 5> getGroups() { return {&means, &quaternions, &scales, &opacities, &colors}; }
    std::array<const std::vector<float>*, 5> getGroups() const { return {&means, &quaternions, &scales, &opacities, &colors}; }

    std::vector<float> means;
    std::vector<float> quaternions;
    std::vector<float> scales;
    std::vector<float> opacities;
    std::vector<float> colors;
};

Gaussians makeRandomGaussians(std::size_t count, Core::Random& random);
Gaussians perturbGaussians(const Gaussians& truth, Core::Random& random);
void clampGaussians(Gaussians& gaussians);

}
