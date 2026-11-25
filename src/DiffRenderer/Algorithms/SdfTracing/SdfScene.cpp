#include "DiffRenderer/Algorithms/SdfTracing/SdfScene.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace DiffRenderer::Algorithms {

using namespace NoorRHI;

namespace {

constexpr float blend = 0.05f;
constexpr float viewDistance = 1.8f;
constexpr float centerJitter = 0.04f;

float smoothMinimum(const float a, const float b) {
    const float h = std::clamp(0.5f + 0.5f * (b - a) / blend, 0.0f, 1.0f);
    return b + (a - b) * h - blend * h * (1.0f - h);
}

SdfSphere jitter(const SdfSphere& sphere, Core::Random& random) {
    return {.center = {sphere.center.x + random.uniform(-centerJitter, centerJitter), sphere.center.y + random.uniform(-centerJitter, centerJitter),
                       sphere.center.z + random.uniform(-centerJitter, centerJitter)},
            .radius = sphere.radius * random.uniform(0.85f, 1.15f)};
}

}

std::vector<float> SdfScene::sampleSpheres(const std::vector<SdfSphere>& spheres) {
    std::vector<float> values;
    values.reserve(gridSize);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t y = 0; y < resolution; ++y)
            for (std::uint32_t x = 0; x < resolution; ++x) {
                const float3 position{(x + 0.5f) / resolution, (y + 0.5f) / resolution, (z + 0.5f) / resolution};
                float distance = 0.0f;
                for (std::size_t index = 0; index < spheres.size(); ++index) {
                    const SdfSphere& sphere = spheres[index];
                    const float3 offset{position.x - sphere.center.x, position.y - sphere.center.y, position.z - sphere.center.z};
                    const float value = std::sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z) - sphere.radius;
                    distance = index == 0 ? value : smoothMinimum(distance, value);
                }
                values.push_back(distance);
            }
    return values;
}

SdfScene::SdfScene(Core::Random& random) {
    const float length = std::sqrt(0.4f * 0.4f + 0.8f * 0.8f + 0.6f * 0.6f);
    lightDirection = {-0.4f / length, 0.8f / length, 0.6f / length};
    const std::vector<SdfSphere> spheres{jitter({{0.42f, 0.48f, 0.5f}, 0.2f}, random), jitter({{0.64f, 0.62f, 0.44f}, 0.13f}, random),
                                         jitter({{0.38f, 0.66f, 0.62f}, 0.09f}, random)};
    grid = sampleSpheres(spheres);
    startGrid = sampleSpheres({{{0.5f, 0.55f, 0.5f}, 0.24f}});
    for (std::size_t view = 0; view < viewCount; ++view) {
        const float azimuth = 2.0f * std::numbers::pi_v<float> * static_cast<float>(view) / viewCount;
        const float elevation = view % 2 ? -0.35f : 0.5f;
        const float3 offset{viewDistance * std::cos(elevation) * std::sin(azimuth), viewDistance * std::sin(elevation),
                            viewDistance * std::cos(elevation) * std::cos(azimuth)};
        views[view].origin = {0.5f + offset.x, 0.5f + offset.y, 0.5f + offset.z};
    }
}

}
