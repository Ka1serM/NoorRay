#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Lights the scene with an equirectangular HDRI (.hdr or .exr), replacing the previous environment map and keeping its strength and rotation.
class EnvironmentImporter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
    static bool isSupported(const std::filesystem::path& path);
};

}
