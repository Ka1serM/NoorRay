#include "DiffRenderer/Algorithms/GaussianSplatting/Gaussians.h"

#include <algorithm>

namespace DiffRenderer::Algorithms {

Gaussians makeRandomGaussians(const std::size_t count, Core::Random& random) {
    Gaussians gaussians(count);
    for (std::size_t index = 0; index < count; ++index) {
        gaussians.means[index * 3] = random.uniform(-1.1f, 1.1f);
        gaussians.means[index * 3 + 1] = random.uniform(-1.1f, 1.1f);
        gaussians.means[index * 3 + 2] = random.uniform(-0.6f, 0.6f);
        for (std::size_t component = 0; component < 4; ++component)
            gaussians.quaternions[index * 4 + component] = random.normal();
        for (std::size_t axis = 0; axis < 3; ++axis) {
            gaussians.scales[index * 3 + axis] = random.uniform(0.02f, 0.12f);
            gaussians.colors[index * 3 + axis] = random.uniform(0.0f, 1.0f);
        }
        gaussians.opacities[index] = random.uniform(0.3f, 0.95f);
    }
    return gaussians;
}

Gaussians perturbGaussians(const Gaussians& truth, Core::Random& random) {
    Gaussians start = truth;
    for (float& value : start.means)
        value += random.uniform(-0.05f, 0.05f);
    for (float& value : start.quaternions)
        value += 0.1f * random.normal();
    for (float& value : start.scales)
        value *= random.uniform(0.8f, 1.25f);
    for (float& value : start.opacities)
        value = std::clamp(value + random.uniform(-0.15f, 0.15f), 0.05f, 0.99f);
    for (float& value : start.colors)
        value += random.uniform(-0.2f, 0.2f);
    return start;
}

void clampGaussians(Gaussians& gaussians) {
    for (float& value : gaussians.scales)
        value = std::max(value, 1e-3f);
    for (float& value : gaussians.opacities)
        value = std::clamp(value, 0.0f, 1.0f);
}

}
