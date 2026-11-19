#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Imports .gltf and .glb files: meshes, PBR materials with the common KHR material extensions, cameras and punctual lights.
class GltfImporter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
};

}
