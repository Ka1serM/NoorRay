#include "DiffRenderer/Core/Adam.h"

#include <cmath>

namespace DiffRenderer::Core {

namespace {

constexpr float beta1 = 0.9f;
constexpr float beta2 = 0.999f;
constexpr float epsilon = 1e-8f;

}

void Adam::reset() {
    first.clear();
    second.clear();
    iteration = 0;
}

void Adam::step(const std::span<float> values, const std::span<const float> gradients) {
    if (first.size() != values.size()) {
        first.assign(values.size(), 0.0f);
        second.assign(values.size(), 0.0f);
        iteration = 0;
    }
    ++iteration;
    const float correction = std::sqrt(1.0f - std::pow(beta2, static_cast<float>(iteration))) / (1.0f - std::pow(beta1, static_cast<float>(iteration)));
    const float rate = learningRate * correction;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float gradient = gradients[index];
        first[index] = beta1 * first[index] + (1.0f - beta1) * gradient;
        second[index] = beta2 * second[index] + (1.0f - beta2) * gradient * gradient;
        values[index] -= rate * first[index] / (std::sqrt(second[index]) + epsilon);
    }
}

}
