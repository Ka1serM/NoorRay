#include "NoorRay/Import/PreviewSurface.h"

#include <cmath>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdShade/shader.h>

namespace NoorRay::Import {

namespace {

struct TextureSource {
    pxr::UsdShadeShader shader;
    pxr::TfToken output;
};

std::optional<TextureSource> findTexture(const pxr::UsdShadeInput& input) {
    if (!input)
        return std::nullopt;
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken outputName;
    pxr::UsdShadeAttributeType type;
    if (!input.GetConnectedSource(&source, &outputName, &type))
        return std::nullopt;
    pxr::UsdShadeShader shader(source.GetPrim());
    pxr::TfToken id;
    if (!shader || !shader.GetShaderId(&id) || id != pxr::TfToken("UsdUVTexture"))
        return std::nullopt;
    return TextureSource{shader, outputName};
}

template <class T>
T getValue(const pxr::UsdShadeShader& shader, const char* name, const T fallback) {
    T value = fallback;
    if (const pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(name)))
        input.Get(&value);
    return value;
}

Scene::UvTransform readUvTransform(const pxr::UsdShadeShader& texture) {
    const pxr::UsdShadeInput st = texture.GetInput(pxr::TfToken("st"));
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken outputName;
    pxr::UsdShadeAttributeType type;
    if (!st || !st.GetConnectedSource(&source, &outputName, &type))
        return {};
    pxr::UsdShadeShader shader(source.GetPrim());
    pxr::TfToken id;
    if (!shader || !shader.GetShaderId(&id) || id != pxr::TfToken("UsdTransform2d"))
        return {};
    const pxr::GfVec2f scale = getValue(shader, "scale", pxr::GfVec2f(1.0f, 1.0f));
    const pxr::GfVec2f translation = getValue(shader, "translation", pxr::GfVec2f(0.0f, 0.0f));
    const float rotation = getValue(shader, "rotation", 0.0f) * Core::pi / 180.0f;
    //! UsdTransform2d scales, then rotates counter-clockwise, then translates, in the bottom-left UV convention.
    const float c = std::cos(rotation);
    const float s = std::sin(rotation);
    Scene::UvTransform transform;
    transform.row0 = {c * scale[0], -s * scale[1], translation[0]};
    transform.row1 = {s * scale[0], c * scale[1], translation[1]};
    return transform;
}

Scene::TextureWrap toWrap(const pxr::TfToken& wrap) {
    if (wrap == "clamp")
        return Scene::TextureWrap::Extend;
    if (wrap == "mirror")
        return Scene::TextureWrap::Mirror;
    if (wrap == "black")
        return Scene::TextureWrap::Clip;
    return Scene::TextureWrap::Repeat;
}

std::uint32_t toChannel(const pxr::TfToken& output) {
    if (output == "g")
        return 1;
    if (output == "b")
        return 2;
    if (output == "a")
        return 3;
    return 0;
}

std::optional<Scene::MaterialTexture> resolveTexture(const TextureSource& source, const Materials::ColorSpace fallbackSpace, const UsdTextureResolver& resolver) {
    pxr::SdfAssetPath file;
    if (const pxr::UsdShadeInput input = source.shader.GetInput(pxr::TfToken("file")))
        input.Get(&file);
    const std::string path = file.GetResolvedPath().empty() ? file.GetAssetPath() : file.GetResolvedPath();
    if (path.empty())
        return std::nullopt;
    Materials::ColorSpace colorSpace = fallbackSpace;
    const pxr::TfToken sourceColorSpace = getValue(source.shader, "sourceColorSpace", pxr::TfToken("auto"));
    if (sourceColorSpace == "raw")
        colorSpace = Materials::ColorSpace::NonColor;
    if (sourceColorSpace == "sRGB")
        colorSpace = Materials::ColorSpace::Srgb;
    std::optional<Scene::MaterialTexture> texture = resolver(path, colorSpace);
    if (!texture)
        return std::nullopt;
    texture->wrap = toWrap(getValue(source.shader, "wrapS", pxr::TfToken("useMetadata")));
    texture->transform = readUvTransform(source.shader);
    return texture;
}

}

std::optional<Scene::Material> PreviewSurface::convert(const pxr::UsdShadeMaterial& material, const UsdTextureResolver& resolver) {
    const pxr::UsdShadeShader surface = material.ComputeSurfaceSource();
    pxr::TfToken id;
    if (!surface || !surface.GetShaderId(&id) || id != pxr::TfToken("UsdPreviewSurface"))
        return std::nullopt;

    Scene::Material result;
    result.name = material.GetPrim().GetName().GetString();
    const auto input = [&](const char* name) { return surface.GetInput(pxr::TfToken(name)); };
    const auto color = [&](const char* name, const pxr::GfVec3f fallback) {
        const pxr::GfVec3f value = getValue(surface, name, fallback);
        return Core::float3{value[0], value[1], value[2]};
    };
    const auto scalarTexture = [&](const char* name, const Materials::ColorSpace space) {
        const std::optional<TextureSource> source = findTexture(input(name));
        if (!source)
            return std::optional<Scene::MaterialTexture>();
        std::optional<Scene::MaterialTexture> texture = resolveTexture(*source, space, resolver);
        if (texture)
            texture->channel = toChannel(source->output);
        return texture;
    };

    result.baseColor = color("diffuseColor", pxr::GfVec3f(0.18f));
    if (const std::optional<TextureSource> source = findTexture(input("diffuseColor"))) {
        result.texture(Scene::TextureSlot::BaseColor) = resolveTexture(*source, Materials::ColorSpace::Srgb, resolver);
        if (result.texture(Scene::TextureSlot::BaseColor))
            result.baseColor = {1.0f, 1.0f, 1.0f};
    }
    result.emissionColor = color("emissiveColor", pxr::GfVec3f(0.0f));
    if (const std::optional<TextureSource> source = findTexture(input("emissiveColor"))) {
        result.texture(Scene::TextureSlot::Emission) = resolveTexture(*source, Materials::ColorSpace::Srgb, resolver);
        if (result.texture(Scene::TextureSlot::Emission))
            result.emissionColor = {1.0f, 1.0f, 1.0f};
    }
    if (getValue(surface, "useSpecularWorkflow", 0) != 0) {
        result.specularTint = color("specularColor", pxr::GfVec3f(0.0f));
        result.metallic = 0.0f;
    } else {
        result.metallic = getValue(surface, "metallic", 0.0f);
        result.texture(Scene::TextureSlot::Metallic) = scalarTexture("metallic", Materials::ColorSpace::NonColor);
        if (result.texture(Scene::TextureSlot::Metallic))
            result.metallic = 1.0f;
    }
    result.roughness = getValue(surface, "roughness", 0.5f);
    result.texture(Scene::TextureSlot::Roughness) = scalarTexture("roughness", Materials::ColorSpace::NonColor);
    if (result.texture(Scene::TextureSlot::Roughness))
        result.roughness = 1.0f;
    result.coat = getValue(surface, "clearcoat", 0.0f);
    result.coatRoughness = getValue(surface, "clearcoatRoughness", 0.01f);
    result.ior = getValue(surface, "ior", 1.5f);
    result.alpha = getValue(surface, "opacity", 1.0f);
    result.texture(Scene::TextureSlot::Alpha) = scalarTexture("opacity", Materials::ColorSpace::NonColor);
    if (result.texture(Scene::TextureSlot::Alpha))
        result.alpha = 1.0f;
    const float threshold = getValue(surface, "opacityThreshold", 0.0f);
    if (result.alpha < 1.0f || result.texture(Scene::TextureSlot::Alpha)) {
        result.alphaMode = threshold > 0.0f ? Scene::AlphaMode::Mask : Scene::AlphaMode::Blend;
        result.alphaCutoff = threshold;
    }
    if (const std::optional<TextureSource> source = findTexture(input("normal")))
        result.texture(Scene::TextureSlot::Normal) = resolveTexture(*source, Materials::ColorSpace::NonColor, resolver);
    return result;
}

}
