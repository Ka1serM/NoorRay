#include "DiffRenderer/Core/Random.h"

#include <cmath>
#include <numbers>

namespace DiffRenderer::Core {

float Random::uniform(const float low, const float high) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const auto bits = static_cast<std::uint32_t>(state >> 40);
    return low + (high - low) * static_cast<float>(bits) / static_cast<float>(1u << 24);
}

float Random::normal() {
    const float radius = std::sqrt(-2.0f * std::log(uniform(1e-7f, 1.0f)));
    return radius * std::cos(2.0f * std::numbers::pi_v<float> * uniform(0.0f, 1.0f));
}

}
