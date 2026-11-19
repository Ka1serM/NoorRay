#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class TextureTable;

//! An alias table over the base level reduced by a power of two to at most maxWidth wide, weighted by mean luminance times sin(theta); empty without a texture.
class EnvironmentBuffers {
public:
    static constexpr std::uint32_t maxWidth = 1024;

    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures);
    void fill(Shaders::SceneRecord& record) const;
    //! The mean radiance over the sphere, for the selection probability against the lights.
    float getAverageLuminance() const { return averageLuminance; }
    bool isPresent() const { return averageLuminance > 0.0f; }

    static Core::float4 readTexel(const Scene::Texture& texture, std::uint32_t x, std::uint32_t y);

private:
    NoorRHI::Buffer<Shaders::AliasEntry> alias;
    NoorRHI::TextureHandle texture;
    NoorRHI::SamplerHandle sampler;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Core::float3 color;
    float strength = 0.0f;
    float rotation = 0.0f;
    bool visibleToCamera = false;
    float averageLuminance = 0.0f;
};

}
