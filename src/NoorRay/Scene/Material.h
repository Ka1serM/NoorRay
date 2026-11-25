#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "NoorRay/Core/Handle.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Materials {
class Graph;
class Program;
struct NodeTree;
}

namespace NoorRay::Scene {

struct Texture;

enum class TextureWrap : std::uint8_t { Repeat, Extend, Clip, Mirror };
enum class TextureFilter : std::uint8_t { Linear, Closest, Cubic };
enum class AlphaMode : std::uint8_t { Opaque, Mask, Blend };
//! The faces of an emitter that light sampling picks, as Cycles' Emission Sampling; Disabled is its None, and Auto samples both faces.
enum class EmissionSampling : std::uint8_t { Auto, Disabled, Front, Back, FrontAndBack };

enum class TextureSlot : std::uint8_t { BaseColor, Alpha, Metallic, Roughness, Normal, Specular, Transmission, Coat, CoatRoughness, CoatNormal, Emission };
inline constexpr std::size_t textureSlotCount = 11;

//! A UV transform in the Blender UV convention: uv' = (dot(row0, uv) + row0.z, dot(row1, uv) + row1.z).
struct UvTransform {
    Core::float3 row0{1.0f, 0.0f, 0.0f};
    Core::float3 row1{0.0f, 1.0f, 0.0f};

    //! KHR_texture_transform and USD UsdTransform2d, given in the top-left origin convention.
    static UvTransform fromTopLeft(Core::float2 offset, float rotation, Core::float2 scale);
    bool operator==(const UvTransform&) const = default;
};

//! channel picks the texture component for a scalar slot; colour and normal slots read rgb.
struct MaterialTexture {
    Core::Handle<Texture> texture;
    TextureWrap wrap = TextureWrap::Repeat;
    TextureFilter filter = TextureFilter::Linear;
    std::uint32_t uvSet = 0;
    std::uint32_t channel = 0;
    UvTransform transform;

    bool operator==(const MaterialTexture&) const = default;
};

//! The fixed Principled BSDF: each input is a factor, times the texture of its slot when that slot is set.
struct Material {
    std::string name;
    Core::float3 baseColor{0.8f, 0.8f, 0.8f};
    bool useVertexColor = false;
    float alpha = 1.0f;
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    float metallic = 0.0f;
    float roughness = 0.5f;
    bool roughnessIsGlossiness = false;
    float normalScale = 1.0f;
    float ior = 1.5f;
    float specularIorLevel = 0.5f;
    Core::float3 specularTint{1.0f, 1.0f, 1.0f};
    float transmission = 0.0f;
    float anisotropic = 0.0f;
    float anisotropicRotation = 0.0f;
    float coat = 0.0f;
    float coatRoughness = 0.03f;
    float coatNormalScale = 1.0f;
    Core::float3 sheenColor;
    float sheenRoughness = 0.5f;
    Core::float3 emissionColor;
    float emissionStrength = 1.0f;
    float thinFilmThickness = 0.0f;
    float thinFilmIor = 1.33f;
    float dispersion = 0.0f;
    EmissionSampling emissionSampling = EmissionSampling::Auto;
    std::array<std::optional<MaterialTexture>, textureSlotCount> textures;
    std::shared_ptr<const Materials::Program> program;
    //! The register values of the program, with the current values of its parameters.
    std::vector<Core::float4> programRegisters;
    //! The node graph that replaces the fixed inputs in the shading; the fixed inputs stay an estimate for the light sampling and the previews.
    std::shared_ptr<const Materials::Graph> graph;
    //! One float4 per parameter slot of the graph.
    std::vector<Core::float4> graphParameters;
    //! The textures of the graph, in graph texture order.
    std::vector<MaterialTexture> graphTextures;
    //! A graph whose emission is the object space offset of a vertex, for a displacement that a program cannot express.
    std::shared_ptr<const Materials::Graph> displacementGraph;
    std::vector<Core::float4> displacementParameters;
    std::vector<MaterialTexture> displacementTextures;
    //! The graph bumps the normal before the displacement.
    bool bumpsRestNormal = false;
    //! The graph bumps the normal of the surface itself.
    bool bumpsSurface = false;
    //! The node tree that the graph, program and displacement compile from.
    std::shared_ptr<const Materials::NodeTree> nodeTree;

    std::optional<MaterialTexture>& texture(const TextureSlot slot) { return textures[std::to_underlying(slot)]; }
    const std::optional<MaterialTexture>& texture(const TextureSlot slot) const { return textures[std::to_underlying(slot)]; }
    bool isEmissive() const { return emissionStrength > 0.0f && (Core::maxComponent(emissionColor) > 0.0f); }
};

}
