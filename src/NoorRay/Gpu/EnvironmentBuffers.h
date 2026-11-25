#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "NoorRHI/Resources/Buffer.h"
#include "NoorRHI/Resources/Image.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"

namespace NoorRay::Gpu {

class GraphBakes;
class TextureTable;

//! An alias table over the base level reduced by a power of two to at most maxWidth wide, weighted by mean luminance times sin(theta); empty without a texture.
class EnvironmentBuffers {
public:
    static constexpr std::uint32_t maxWidth = 1024;

    [[nodiscard]] std::expected<void, Core::Error> update(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures, GraphBakes& bakes);
    void fill(Shaders::SceneRecord& record) const;
    //! The mean radiance over the sphere, for the selection probability against the lights.
    float getAverageLuminance() const { return averageLuminance; }
    std::uint32_t getHidden() const { return hidden; }
    bool isPresent() const { return averageLuminance > 0.0f; }

    static Core::float4 readTexel(const Scene::Texture& texture, std::uint32_t x, std::uint32_t y);

private:
    [[nodiscard]] std::expected<void, Core::Error> bake(NoorRHI::Device& device, const Scene::Scene& scene, const TextureTable& textures, GraphBakes& bakes);

    NoorRHI::Buffer<Shaders::AliasEntry> alias;
    NoorRHI::TextureHandle texture;
    //! The baked maps of a world graph: for all rays, and for camera rays when the graph reads the ray type.
    Scene::Texture baked;
    NoorRHI::Image image;
    NoorRHI::Image cameraImage;
    std::uint64_t bakedKey = 0;
    NoorRHI::SamplerHandle sampler;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Core::float3 color;
    float strength = 0.0f;
    float rotation = 0.0f;
    std::uint32_t hidden = 0;
    float cameraView = Shaders::environmentTransparent;
    float averageLuminance = 0.0f;
};

}
