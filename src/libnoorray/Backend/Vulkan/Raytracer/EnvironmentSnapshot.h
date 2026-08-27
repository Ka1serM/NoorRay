#pragma once

#include <cstdint>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

class Environment;

// Immutable per-frame view of the scene environment, uploaded into a
// descriptor-heap storage buffer. The layout mirrors EnvironmentRecord in
// Shaders/Raytracer/Environment.slang; keep the two in step.
struct VulkanEnvironmentSnapshot
{
    static constexpr std::uint32_t NoTexture = 0xFFFFFFFFu;

    glm::vec3 color{1.0f};
    float rotationSin{};

    float rotationCos{1.0f};
    float visibleExposureScale{1.0f};
    float lightingExposureScale{1.0f};
    float importanceWeight{};

    std::uint32_t texture{NoTexture};
    std::uint32_t cdfTexture{NoTexture};
    std::int32_t cdfWidth{};
    std::int32_t cdfHeight{};

    std::int32_t mapping{};
    std::int32_t padding0{};
    std::int32_t padding1{};
    std::int32_t padding2{};

    // std430 lays a float3x3 out as three 16-byte-aligned columns.
    glm::vec4 environmentFromWorld[3]{};
};

// Fills a snapshot from the scene environment. `texture` and `cdfTexture` are
// descriptor-heap indices the caller has already published, or NoTexture.
VulkanEnvironmentSnapshot makeEnvironmentSnapshot(const Environment& environment,
    std::uint32_t texture, std::uint32_t cdfTexture);
