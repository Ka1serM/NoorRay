#include "material.h"

#include "renderParam.h"

#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/warning.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/usd/sdf/assetPath.h>

#include "Shading/Sellmeier.h"

#include <algorithm>
#include <mutex>

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
float FloatValue(const VtValue& value, const float fallback)
{
    if (value.IsHolding<float>())
        return value.UncheckedGet<float>();
    if (value.IsHolding<double>())
        return static_cast<float>(value.UncheckedGet<double>());
    return fallback;
}

glm::vec3 ColorValue(const VtValue& value, const glm::vec3 fallback)
{
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f& v = value.UncheckedGet<GfVec3f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3d& v = value.UncheckedGet<GfVec3d>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec4f>()) {
        const GfVec4f& v = value.UncheckedGet<GfVec4f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec4d>()) {
        const GfVec4d& v = value.UncheckedGet<GfVec4d>();
        return {
            static_cast<float>(v[0]), static_cast<float>(v[1]),
            static_cast<float>(v[2])};
    }
    return fallback;
}

std::string TextureFilePath(const VtValue& value)
{
    if (value.IsHolding<SdfAssetPath>()) {
        const SdfAssetPath& assetPath = value.UncheckedGet<SdfAssetPath>();
        std::string path = assetPath.GetResolvedPath();
        if (path.empty())
            path = assetPath.GetAssetPath();
        return path;
    }
    if (value.IsHolding<std::string>())
        return value.UncheckedGet<std::string>();
    return {};
}

void ApplyPreviewSurface(
    const std::map<TfToken, VtValue>& parameters, Material& material)
{
    const auto get = [&](const char* name) -> const VtValue* {
        const auto found = parameters.find(TfToken(name));
        return found == parameters.end() ? nullptr : &found->second;
    };
    if (const VtValue* v = get("diffuseColor"))
        material.albedo = ColorValue(*v, material.albedo);
    if (const VtValue* v = get("roughness"))
        material.roughness = std::clamp(FloatValue(*v, material.roughness), 0.0f, 1.0f);
    if (const VtValue* v = get("metallic"))
        material.metallic = std::clamp(FloatValue(*v, material.metallic), 0.0f, 1.0f);
    if (const VtValue* v = get("opacity"))
        material.opacity = std::clamp(FloatValue(*v, material.opacity), 0.0f, 1.0f);
    if (const VtValue* v = get("transmission"))
        material.transmission = std::clamp(FloatValue(*v, material.transmission), 0.0f, 1.0f);
    if (const VtValue* v = get("transmissionColor"))
        material.transmissionColor = ColorValue(*v, material.transmissionColor);
    if (const VtValue* v = get("ior"))
        material.sellmeier = constantIorSellmeier(FloatValue(*v, 1.5f));
    if (const VtValue* v = get("emissiveColor")) {
        material.emission = ColorValue(*v, material.emission);
        material.emissionStrength = std::max({
            material.emission.x, material.emission.y, material.emission.z});
        if (material.emissionStrength > 0.0f)
            material.emission /= material.emissionStrength;
    }
    if (const VtValue* v = get("specularColor")) {
        const glm::vec3 color = ColorValue(*v, glm::vec3(material.specular));
        material.specular = std::clamp(
            (color.x + color.y + color.z) / 3.0f, 0.0f, 1.0f);
    }
}

void ApplyTextureConnection(
    const TfToken& inputName,
    const int textureIndex,
    Material& material)
{
    const std::string& name = inputName.GetString();
    if (name == "diffuseColor")
        material.albedoIndex = textureIndex;
    else if (name == "roughness")
        material.roughnessIndex = textureIndex;
    else if (name == "metallic")
        material.metallicIndex = textureIndex;
    else if (name == "specularColor")
        material.specularIndex = textureIndex;
    else if (name == "opacity")
        material.opacityIndex = textureIndex;
    else if (name == "transmission")
        material.transmissionIndex = textureIndex;
    else if (name == "emissiveColor")
        material.emissionIndex = textureIndex;
    else if (name == "normal")
        material.normalIndex = textureIndex;
}
}

HdNoorRayMaterial::HdNoorRayMaterial(const SdfPath& id)
    : HdMaterial(id)
{
    material_.albedo = glm::vec3(0.8f);
    material_.roughness = 0.5f;
}

void HdNoorRayMaterial::Sync(
    HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    const VtValue resource = sceneDelegate->GetMaterialResource(GetId());
    Material parsed;
    parsed.albedo = glm::vec3(0.8f);
    parsed.roughness = 0.5f;
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    if (resource.IsHolding<HdMaterialNetworkMap>()) {
        const HdMaterialNetworkMap& networks =
            resource.UncheckedGet<HdMaterialNetworkMap>();
        for (const auto& [terminal, network] : networks.map) {
            (void)terminal;
            for (const HdMaterialNode& node : network.nodes) {
                if (node.identifier.GetString().find("UsdPreviewSurface")
                    != std::string::npos)
                    ApplyPreviewSurface(node.parameters, parsed);
            }
        }
    } else if (resource.IsHolding<HdMaterialNetwork2>()) {
        const HdMaterialNetwork2& network =
            resource.UncheckedGet<HdMaterialNetwork2>();

        // First pass: collect texture nodes and load textures.
        std::map<SdfPath, int> textureNodeIndices;
        for (const auto& [path, node] : network.nodes) {
            if (node.nodeTypeId.GetString().find("UsdUVTexture")
                == std::string::npos)
                continue;
            const auto fileIt = node.parameters.find(TfToken("file"));
            if (fileIt == node.parameters.end())
                continue;
            const std::string filePath = TextureFilePath(fileIt->second);
            if (filePath.empty())
                continue;
            try {
                const int texIndex = param.GetOrCreateTexture(
                    filePath, TextureEncoding::Linear8);
                textureNodeIndices[path] = texIndex;
            } catch (const std::exception& e) {
                TF_WARN("Failed to load texture '%s': %s",
                    filePath.c_str(), e.what());
            }
        }

        // Second pass: process UsdPreviewSurface and wire up textures.
        for (const auto& [path, node] : network.nodes) {
            if (node.nodeTypeId.GetString().find("UsdPreviewSurface")
                == std::string::npos)
                continue;
            ApplyPreviewSurface(node.parameters, parsed);
            for (const auto& [inputName, connections] : node.inputConnections) {
                if (connections.empty())
                    continue;
                const auto texIt = textureNodeIndices.find(
                    connections[0].upstreamNode);
                if (texIt == textureNodeIndices.end())
                    continue;
                ApplyTextureConnection(inputName, texIt->second, parsed);
            }
        }
    }
    material_ = parsed;
    std::scoped_lock lock(param.mutex);
    param.PublishMaterial(GetId(), material_);
    *dirtyBits = Clean;
}

HdDirtyBits HdNoorRayMaterial::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

PXR_NAMESPACE_CLOSE_SCOPE
