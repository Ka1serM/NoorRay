#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Imports USD stages on the desktop: meshes with GeomSubsets, UsdPreviewSurface materials, cameras and UsdLux lights.
class UsdImporter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
};

}
