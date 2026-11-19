#pragma once

#include <functional>
#include <optional>
#include <string>

#include <pxr/usd/usdShade/material.h>

#include "NoorRay/Materials/TextureCache.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Resolves a texture file path to a scene texture in the given colour space; nullopt skips the texture.
using UsdTextureResolver = std::function<std::optional<Scene::MaterialTexture>(const std::string& path, Materials::ColorSpace colorSpace)>;

//! Converts a material whose surface is a UsdPreviewSurface network; other surfaces return nullopt.
class PreviewSurface {
public:
    static std::optional<Scene::Material> convert(const pxr::UsdShadeMaterial& material, const UsdTextureResolver& resolver);
};

}
