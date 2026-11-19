#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Import {

//! Chooses the importer by file extension and adds the file's contents to the scene.
class SceneImporter {
public:
    [[nodiscard]] static std::expected<void, Core::Error> import(const std::filesystem::path& path, Scene::Scene& scene);
    static bool isSupported(const std::filesystem::path& path);
};

}
