#include "NoorRay/Import/GltfMaterial.h"

#include <algorithm>
#include <numbers>

#include <tiny_gltf.h>

namespace NoorRay::Import {

namespace {

using Materials::ColorSpace;

const tinygltf::Value* findExtension(const tinygltf::ExtensionMap& extensions, const char* name) {
    const auto found = extensions.find(name);
    return found == extensions.end() ? nullptr : &found->second;
}

float getNumber(const tinygltf::Value& object, const char* key, const float fallback) {
    if (!object.Has(key) || !object.Get(key).IsNumber())
        return fallback;
    return static_cast<float>(object.Get(key).GetNumberAsDouble());
}

Core::float3 getColor(const tinygltf::Value& object, const char* key, const Core::float3 fallback) {
    if (!object.Has(key))
        return fallback;
    const tinygltf::Value& value = object.Get(key);
    if (!value.IsArray() || value.ArrayLen() < 3)
        return fallback;
    return {static_cast<float>(value.Get(0).GetNumberAsDouble()), static_cast<float>(value.Get(1).GetNumberAsDouble()),
            static_cast<float>(value.Get(2).GetNumberAsDouble())};
}

Scene::UvTransform readTransform(const tinygltf::ExtensionMap& extensions, int& texCoord) {
    const tinygltf::Value* transform = findExtension(extensions, "KHR_texture_transform");
    if (!transform)
        return {};
    Core::float2 offset{0.0f, 0.0f};
    Core::float2 scale{1.0f, 1.0f};
    if (transform->Has("offset") && transform->Get("offset").ArrayLen() == 2)
        offset = {static_cast<float>(transform->Get("offset").Get(0).GetNumberAsDouble()),
                  static_cast<float>(transform->Get("offset").Get(1).GetNumberAsDouble())};
    if (transform->Has("scale") && transform->Get("scale").ArrayLen() == 2)
        scale = {static_cast<float>(transform->Get("scale").Get(0).GetNumberAsDouble()),
                 static_cast<float>(transform->Get("scale").Get(1).GetNumberAsDouble())};
    if (transform->Has("texCoord") && transform->Get("texCoord").IsInt())
        texCoord = transform->Get("texCoord").GetNumberAsInt();
    return Scene::UvTransform::fromTopLeft(offset, getNumber(*transform, "rotation", 0.0f), scale);
}

std::optional<Scene::MaterialTexture> resolve(const GltfTextureResolver& resolver, const int index, int texCoord, const tinygltf::ExtensionMap& extensions,
                                  const ColorSpace colorSpace) {
    if (index < 0)
        return std::nullopt;
    std::optional<Scene::MaterialTexture> texture = resolver(index, colorSpace);
    if (!texture)
        return std::nullopt;
    texture->transform = readTransform(extensions, texCoord);
    texture->uvSet = static_cast<std::uint32_t>(std::clamp(texCoord, 0, static_cast<int>(Scene::maxUvSets) - 1));
    return texture;
}

void setChannel(std::optional<Scene::MaterialTexture>& texture, const std::uint32_t channel) {
    if (texture)
        texture->channel = channel;
}

std::optional<Scene::MaterialTexture> resolveInfo(const GltfTextureResolver& resolver, const tinygltf::Value& object, const char* key, const ColorSpace colorSpace) {
    if (!object.Has(key))
        return std::nullopt;
    const tinygltf::Value& info = object.Get(key);
    const int index = info.Has("index") ? info.Get("index").GetNumberAsInt() : -1;
    const int texCoord = info.Has("texCoord") ? info.Get("texCoord").GetNumberAsInt() : 0;
    tinygltf::ExtensionMap extensions;
    if (info.Has("extensions") && info.Get("extensions").IsObject())
        for (const std::string& name : info.Get("extensions").Keys())
            extensions.emplace(name, info.Get("extensions").Get(name));
    return resolve(resolver, index, texCoord, extensions, colorSpace);
}

}

Scene::Material GltfMaterial::convert(const tinygltf::Material& material, const GltfTextureResolver& resolver) {
    Scene::Material result;
    result.name = material.name;
    const tinygltf::PbrMetallicRoughness& pbr = material.pbrMetallicRoughness;
    result.baseColor = {static_cast<float>(pbr.baseColorFactor[0]), static_cast<float>(pbr.baseColorFactor[1]), static_cast<float>(pbr.baseColorFactor[2])};
    result.alpha = static_cast<float>(pbr.baseColorFactor[3]);
    result.texture(Scene::TextureSlot::BaseColor) = resolve(resolver, pbr.baseColorTexture.index, pbr.baseColorTexture.texCoord, pbr.baseColorTexture.extensions, ColorSpace::Srgb);
    result.metallic = static_cast<float>(pbr.metallicFactor);
    result.roughness = static_cast<float>(pbr.roughnessFactor);
    result.texture(Scene::TextureSlot::Metallic) = resolve(resolver, pbr.metallicRoughnessTexture.index, pbr.metallicRoughnessTexture.texCoord,
                                     pbr.metallicRoughnessTexture.extensions, ColorSpace::NonColor);
    result.texture(Scene::TextureSlot::Roughness) = result.texture(Scene::TextureSlot::Metallic);
    setChannel(result.texture(Scene::TextureSlot::Metallic), 2);
    setChannel(result.texture(Scene::TextureSlot::Roughness), 1);
    result.texture(Scene::TextureSlot::Normal) =
        resolve(resolver, material.normalTexture.index, material.normalTexture.texCoord, material.normalTexture.extensions, ColorSpace::NonColor);
    result.normalScale = static_cast<float>(material.normalTexture.scale);
    result.emissionColor = {static_cast<float>(material.emissiveFactor[0]), static_cast<float>(material.emissiveFactor[1]),
                            static_cast<float>(material.emissiveFactor[2])};
    result.texture(Scene::TextureSlot::Emission) =
        resolve(resolver, material.emissiveTexture.index, material.emissiveTexture.texCoord, material.emissiveTexture.extensions, ColorSpace::Srgb);
    if (result.texture(Scene::TextureSlot::Emission) && Core::maxComponent(result.emissionColor) == 0.0f)
        result.texture(Scene::TextureSlot::Emission).reset();

    if (material.alphaMode == "MASK")
        result.alphaMode = Scene::AlphaMode::Mask;
    if (material.alphaMode == "BLEND")
        result.alphaMode = Scene::AlphaMode::Blend;
    result.alphaCutoff = static_cast<float>(material.alphaCutoff);
    if (result.alphaMode != Scene::AlphaMode::Opaque)
        result.texture(Scene::TextureSlot::Alpha) = result.texture(Scene::TextureSlot::BaseColor);
    setChannel(result.texture(Scene::TextureSlot::Alpha), 3);

    const tinygltf::ExtensionMap& extensions = material.extensions;
    if (const tinygltf::Value* strength = findExtension(extensions, "KHR_materials_emissive_strength"))
        result.emissionStrength = getNumber(*strength, "emissiveStrength", 1.0f);
    if (const tinygltf::Value* ior = findExtension(extensions, "KHR_materials_ior"))
        result.ior = getNumber(*ior, "ior", 1.5f);
    if (const tinygltf::Value* specular = findExtension(extensions, "KHR_materials_specular")) {
        result.specularIorLevel = 0.5f * getNumber(*specular, "specularFactor", 1.0f);
        result.texture(Scene::TextureSlot::Specular) = resolveInfo(resolver, *specular, "specularTexture", ColorSpace::NonColor);
        setChannel(result.texture(Scene::TextureSlot::Specular), 3);
        result.specularTint = getColor(*specular, "specularColorFactor", {1.0f, 1.0f, 1.0f});
    }
    if (const tinygltf::Value* transmission = findExtension(extensions, "KHR_materials_transmission")) {
        result.transmission = getNumber(*transmission, "transmissionFactor", 0.0f);
        result.texture(Scene::TextureSlot::Transmission) = resolveInfo(resolver, *transmission, "transmissionTexture", ColorSpace::NonColor);
    }
    if (const tinygltf::Value* clearcoat = findExtension(extensions, "KHR_materials_clearcoat")) {
        result.coat = getNumber(*clearcoat, "clearcoatFactor", 0.0f);
        result.texture(Scene::TextureSlot::Coat) = resolveInfo(resolver, *clearcoat, "clearcoatTexture", ColorSpace::NonColor);
        result.coatRoughness = getNumber(*clearcoat, "clearcoatRoughnessFactor", 0.0f);
        result.texture(Scene::TextureSlot::CoatRoughness) = resolveInfo(resolver, *clearcoat, "clearcoatRoughnessTexture", ColorSpace::NonColor);
        setChannel(result.texture(Scene::TextureSlot::CoatRoughness), 1);
        result.texture(Scene::TextureSlot::CoatNormal) = resolveInfo(resolver, *clearcoat, "clearcoatNormalTexture", ColorSpace::NonColor);
        if (clearcoat->Has("clearcoatNormalTexture"))
            result.coatNormalScale = getNumber(clearcoat->Get("clearcoatNormalTexture"), "scale", 1.0f);
    }
    if (const tinygltf::Value* sheen = findExtension(extensions, "KHR_materials_sheen")) {
        result.sheenColor = getColor(*sheen, "sheenColorFactor", {});
        result.sheenRoughness = getNumber(*sheen, "sheenRoughnessFactor", 0.0f);
    }
    if (const tinygltf::Value* anisotropy = findExtension(extensions, "KHR_materials_anisotropy")) {
        result.anisotropic = getNumber(*anisotropy, "anisotropyStrength", 0.0f);
        //! Blender measures the rotation in turns; the flip to the bottom-left UV origin turns it the other way.
        result.anisotropicRotation = -getNumber(*anisotropy, "anisotropyRotation", 0.0f) / (2.0f * std::numbers::pi_v<float>);
    }
    if (const tinygltf::Value* iridescence = findExtension(extensions, "KHR_materials_iridescence")) {
        result.thinFilmThickness = getNumber(*iridescence, "iridescenceFactor", 0.0f) * getNumber(*iridescence, "iridescenceThicknessMaximum", 400.0f);
        result.thinFilmIor = getNumber(*iridescence, "iridescenceIor", 1.3f);
    }
    if (const tinygltf::Value* dispersion = findExtension(extensions, "KHR_materials_dispersion"))
        result.dispersion = getNumber(*dispersion, "dispersion", 0.0f);
    if (findExtension(extensions, "KHR_materials_unlit")) {
        result.emissionColor = result.baseColor;
        result.texture(Scene::TextureSlot::Emission) = result.texture(Scene::TextureSlot::BaseColor);
        result.emissionStrength = 1.0f;
        result.baseColor = {};
        result.texture(Scene::TextureSlot::BaseColor).reset();
        result.roughness = 1.0f;
        result.specularIorLevel = 0.0f;
    }
    return result;
}

}
